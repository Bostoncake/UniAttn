import sys
import numpy as np
import time


def pseudoinverse(B):
    # Perform Singular Value Decomposition
    U, sigma, VT = np.linalg.svd(B, full_matrices=False)
    
    sigma_plus = np.zeros_like(sigma)
    non_zero_indices = sigma > 1e-10  # threshold to handle numerical stability
    sigma_plus[non_zero_indices] = 1 / sigma[non_zero_indices]
    Sigma_plus = np.diag(sigma_plus)

    B_plus = VT.T @ Sigma_plus @ U.T
    return B_plus

index = int(sys.argv[1])
base_dir = str(sys.argv[2])
group_idx = int(sys.argv[3])
group_begin = int(sys.argv[4])
group_end = int(sys.argv[5])
total_layer = int(sys.argv[6])

o1 = np.load(f"{base_dir}/result_base_output.npy")
o2 = np.load(f"{base_dir}/result_sequential_index_{index}.npy")

X = np.load(f"{base_dir}/all_hidden_states_v2_sequential_index_{index}.npy")

print(o1.shape)
print(o2.shape)
print(X.shape)

a = o1 - o2
b = X

a_list = []
b_list = []

counter = 0
for i in range(total_layer):
    if i<group_begin or (i-group_begin)%group_idx==0 or i>=group_end:
        pass
    else:
        if counter < index + 1:
            print(i)
            a_list.append(a[i])
            b_list.append(b[i])
            counter += 1
        else:
            break

init_weight_list = []
for i in [index]:
    c = np.matmul(pseudoinverse(b_list[i]), a_list[i])
    init_weight_list.append(c)
    print(np.linalg.norm(a_list[i] - np.matmul(b_list[i], c)))

save_index = index + 1
np.save(f'{base_dir}/init_weights_sequential_index_{save_index}.npy', np.array(init_weight_list))