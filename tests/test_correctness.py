from types import SimpleNamespace

import numpy as np
import pytest
import torch

from compare_runs import mean_ci
from main import train, evaluate
from models.agent import DDPGAgent
from models.ppo_agent import PPOAgent, compute_gae
from models.replay_buffer import ReplayBuffer
from models.rollout_buffer import RolloutBuffer


def column(values):
    return torch.tensor(values, dtype=torch.float32).reshape(-1, 1)


def test_gae_does_not_cross_episode_boundaries():
    for next_episode_reward in (100., 200.):
        _, returns = compute_gae(column([1, 1, next_episode_reward]), column([0, 0, 0]),
                                 column([0, 0, 0]), column([0, 1, 0]), column([0, 1, 0]), .99, .95)
        torch.testing.assert_close(returns[:2], column([1.9405, 1]))


def test_gae_timeout_bootstraps_final_observation_and_stops_recursion():
    _, returns = compute_gae(column([1, 100]), column([2, 999]), column([10, 0]),
                             column([0, 1]), column([1, 1]), .99, .95)
    torch.testing.assert_close(returns, column([10.9, 100]))


def test_gae_true_terminal_ignores_next_value():
    _, returns = compute_gae(column([1]), column([2]), column([999]), column([1]), column([1]), .99, .95)
    torch.testing.assert_close(returns, column([1]))


def test_gae_rollout_cut_bootstraps():
    _, returns = compute_gae(column([1]), column([2]), column([10]), column([0]), column([0]), .99, .95)
    torch.testing.assert_close(returns, column([10.9]))


def test_ddpg_targets_and_soft_updates():
    agent = DDPGAgent(1, 1, 1., lr=0., device="cpu")
    for network in (agent.critic, agent.critic_target):
        for param in network.parameters():
            param.data.zero_()
    agent.critic_target.net[-1].bias.data.fill_(10.)
    class FixedBuffer:
        def sample(self, size):
            return column([0, 0]), column([0, 0]), column([1, 1]), column([0, 0]), column([0, 1])
    critic_loss, _ = agent.train(FixedBuffer(), 2)
    assert critic_loss == pytest.approx((10.9**2 + 1)/2, rel=1e-6)
    assert agent.critic_target.net[-1].bias.item() == pytest.approx(9.95)


def test_single_transition_ppo_update_is_finite():
    agent = PPOAgent(1, 1, 1., ppo_epochs=1, device="cpu")
    buffer = RolloutBuffer(1, 1, 1, "cpu")
    action, logprob, value = agent.select_action(np.array([0.], dtype=np.float32))
    buffer.push([0], action, logprob, 1., value, [1], False, False)
    assert np.isfinite(agent.train(buffer)).all()
    assert buffer.ptr == 0


class FakeEnv:
    observation_space = SimpleNamespace(shape=(1,))
    action_space = SimpleNamespace(shape=(1,))
    def reset(self, seed=None):
        self.episode_step = 0
        return np.array([-100.], dtype=np.float32), {}
    def step(self, action):
        self.episode_step += 1
        return np.array([self.episode_step], dtype=np.float32), 1., False, self.episode_step == 2, {}


class NullWriter:
    def add_scalar(self, *args):
        pass


@pytest.mark.parametrize("algo", ["ddpg", "ppo"])
def test_training_budget_and_timeout_storage(algo):
    calls = []
    class Agent:
        device = "cpu"
        max_action = 1.
        def select_action(self, state, **kwargs):
            if algo == "ddpg":
                return np.array([0.])
            return np.array([2.]), np.array([-1.]), 0.
        def train(self, buffer, *args):
            if algo == "ddpg":
                assert not buffer.dones.any()
            else:
                assert buffer.ptr in (4, 1)
                assert (buffer.next_states[:buffer.ptr] > 0).all()
                assert (buffer.actions[:buffer.ptr] == 2).all()  # Keep raw policy actions.
                assert not buffer.terminated.any()
                if buffer.ptr == 4:
                    torch.testing.assert_close(buffer.episode_ends, column([0, 1, 0, 1]))
                calls.append(buffer.ptr)
                buffer.clear()
            return 0., 0.
    args = SimpleNamespace(algo=algo, capacity=10, ppo_steps=4, seed=1, total_steps=5,
                           batch_size=2, noise_scale=.1, eval_interval=2, run_id=1)
    checkpoints = []
    train(Agent(), args, FakeEnv(), NullWriter(), checkpoints.append)
    assert checkpoints == [2, 4, 5]
    if algo == "ppo":
        assert calls == [4, 1]


def test_evaluation_preserves_training_rng():
    agent = DDPGAgent(1, 1, 1., device="cpu")
    np.random.seed(42)
    expected = np.random.random(3)
    np.random.seed(42)
    before = torch.get_rng_state().clone()
    assert evaluate(agent, "ddpg", FakeEnv(), [10, 11]) == [2., 2.]
    np.testing.assert_array_equal(np.random.random(3), expected)
    assert torch.equal(before, torch.get_rng_state())


def test_confidence_interval_uses_actual_sample_size():
    mean, ci = mean_ci([1., 3.])
    assert mean == 2.
    assert ci == pytest.approx(12.7062047364)
    assert np.isnan(mean_ci([1.])[1])
