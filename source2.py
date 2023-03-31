import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mohr_circle3d import plot_mohr3d
from stress_invariants import stress_invariants

# Chỉnh in ra 4 chữ số sau dấu phẩy
np.set_printoptions(precision=4, suppress=True)

# Input tensor ứng suất
stress_array = np.array([[4.0, 3.0, 5.0], 
                         [3.0, 0.0, 3.0], 
                         [5.0, 3.0, 2.0]])
stress_tensor = np.reshape(stress_array, (3, 3))

# Hằng số đàn hồi
E = 200.0
v = 0.4
def source2_calc(stress_tensor, E, v):
    # Các bất biến của tensor ứng suất
    I1, I2, I3, dev_stress_tensor = stress_invariants(stress_tensor)

    # Các ứng suất chính
    eig_vals, eig_vecs = np.linalg.eig(stress_tensor)
    idx = np.argsort(eig_vals)[::-1] # lấy chỉ số theo thứ tự giảm dần
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[:, idx]
    o1, o2, o3 = eig_vals

    # Các phương chính ứng suất
    p1, p2, p3 = eig_vecs[:, idx[0]], eig_vecs[:, idx[1]], eig_vecs[:, idx[2]]
    
    # Vẽ đồ thị vòng tròn Mohr ứng suất
    # plt.figure()
    # plot_mohr3d(stress_tensor)

    # Tính ứng suất thủy tĩnh
    hydro_stress = I1 / 3.0

    # Tensor ứng suất cầu
    spherical_stress_tensor = np.eye(3) * hydro_stress

    # Tensor ứng suất lệch
    dev_spherical_stress_tensor = stress_tensor - spherical_stress_tensor

    # Xác định tensor biến dạng
    strain_tensor = 1 / (2 * E * (1 + v)) * dev_stress_tensor + \
                    v / (E * (1 - 2 * v)) * np.trace(dev_stress_tensor) * np.eye(3)

    return I1, I2, I3, o1, o2, o3, p1, p2, p3, hydro_stress, spherical_stress_tensor, dev_spherical_stress_tensor, strain_tensor
    
# In kết quả
# print("Các bất biến của tensor ứng suất:")
# print("I1 =", f"{I1:.4f}")
# print("I2 =", f"{I2:.4f}")
# print("I3 =", f"{I3:.4f}")
I1, I2, I3, o1, o2, o3, p1, p2, p3, hydro_stress, spherical_stress_tensor, dev_spherical_stress_tensor, strain_tensor = source2_calc(stress_tensor, E, v)
print("\nCác ứng suất chính:")
print("o1 =", f"{o1:.4f}")
print("o2 =", f"{o2:.4f}")
print("o3 =", f"{o3:.4f}")
print("\nCác phương chính ứng suất:")
print("p1 =", p1)
print("p2 =", p2)
print("p3 =", p3)
print("\nTensor ứng suất thủy tĩnh:")
print(f"{hydro_stress:.4f}")
print("\nTensor ứng suất cầu:")
print(spherical_stress_tensor)
print("\nTensor ứng suất lệch:")
print(dev_spherical_stress_tensor)
print("\nTensor biến dạng:")
print(strain_tensor)