import subprocess
import concurrent.futures
import os
import time

def run_experiment(algo, run_id):
    print(f"Starting {algo} run {run_id}")
    log_dir = f"runs/{algo}_halfcheetah_run_{run_id}"
    
    # Check if run already exists and has 1000 episodes
    # In this case we just run it regardless or skip if it exists? 
    # For now, let's just run it. We might want to clear runs dir if we are restarting.
    cmd = [
        "conda", "run", "-n", "aloc",
        "python", "main.py",
        "--algo", algo,
        "--run_id", str(run_id),
        "--max_episodes", "1000"
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"Completed {algo} run {run_id}", flush=True)
    except subprocess.CalledProcessError as e:
        print(f"Error in {algo} run {run_id}: {e}", flush=True)

if __name__ == '__main__':
    algos = ["ddpg", "ppo"]
    runs = 20
    
    start_time = time.time()
    
    tasks = []
    # Using a process pool or thread pool
    # The algorithms might use the GPU or CPU heavily, so let's limit max_workers to avoid crashing the system or taking forever
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        for algo in algos:
            for i in range(1, runs + 1):
                tasks.append(executor.submit(run_experiment, algo, i))
                
        concurrent.futures.wait(tasks)
        
    print(f"All experiments completed in {time.time() - start_time:.2f} seconds")
