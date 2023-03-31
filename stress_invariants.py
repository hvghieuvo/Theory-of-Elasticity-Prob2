import numpy as np

def stress_invariants(stress_tensor):
    I1 = np.trace(stress_tensor)
    # I2 = np.trace(np.dot(stress_tensor, stress_tensor)) - np.trace(stress_tensor)**2 / 3
    I2 = 0.5 * (np.trace(stress_tensor)**2 - np.trace(np.dot(stress_tensor, stress_tensor)))
    I3 = np.linalg.det(stress_tensor)
    p = I1 / 3.0
    dev_stress_tensor = stress_tensor - p * np.eye(3)

    return I1, I2, I3, dev_stress_tensor
