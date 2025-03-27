import subprocess
import os
import time

SCRIPT_NAME = "train_normal_NN_BatchNorm.py"
TOTAL_JOBS = 20
MAX_PARALLEL = min(4, os.cpu_count())  # Change 4 to desired parallelism

running = []

for i in range(TOTAL_JOBS):
    print(f"🚀 Launching iteration {i+1}")
    proc = subprocess.Popen(
        ["python", SCRIPT_NAME, "--iteration", str(i)],
        env=dict(os.environ, OPENBLAS_NUM_THREADS="1")  # Limit threading
    )
    running.append((i, proc))

    # Control concurrency
    while len(running) >= MAX_PARALLEL:
        for j, p in running:
            if p.poll() is not None:  # Done
                out, err = p.communicate()
                print(f"✅ Iteration {j+1} completed.")
                if err:
                    print(f"⚠️ Error in iteration {j+1}: {err.decode()}")
                running.remove((j, p))
                break
        else:
            time.sleep(1)

# Wait for the last few
for j, p in running:
    p.wait()
    out, err = p.communicate()
    print(f"✅ Iteration {j+1} completed.")
    if err:
        print(f"⚠️ Error in iteration {j+1}: {err.decode()}")

print("🎉 All 20 jobs finished.")
