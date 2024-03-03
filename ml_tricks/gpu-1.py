import numpy as np
import datetime
from cupyx.profiler import benchmark
import cupy as cp

t1 = datetime.datetime.now()
size = 4096 * 4096
input = np.random.random(size).astype(np.float32)
t2 = datetime.datetime.now()

print(f"time execution in CPU = {t2-t1}")


input_gpu = cp.asarray(input)
execution_gpu = benchmark(cp.sort, (input_gpu,), n_repeat=10)
gpu_avg_time = np.average(execution_gpu.gpu_times)
print(f"{gpu_avg_time:.6f} s")
