import numpy as np
import matplotlib.pyplot as plt

x = np.array([0, 1.3, 2.1, 2.8, 4.2, 5.7])
y = np.array([-1, -1, -1, 1, -1, 1])
t_vals = np.linspace(0, 5, 100)
J_vals = []

for t in t_vals:
    z = x - t
    epsilon = np.maximum(0, 1 - y*z)
    J_vals.append(np.sum(epsilon))

plt.figure(figsize=(8, 6))
plt.plot(t_vals, J_vals)
plt.xlabel('t')
plt.ylabel('J(t)')
plt.title('Hinge Loss vs Threshold')
plt.grid(True)
plt.tight_layout()
plt.savefig('problem2_hinge_loss.png', dpi=150, bbox_inches='tight')
plt.close()

