import numpy as np
import time
import sys

max_i = int(sys.argv[1])
base_dir = str(sys.argv[2])

o3 = []
for i in range(1,max_i+1):
    o3.append(np.load(f"{base_dir}/init_weights_sequential_index_{i}.npy")[0,:,:])
result = np.stack(o3, axis=0)

np.save(f'{base_dir}/init_weights_sequential_new_plan.npy',result)
