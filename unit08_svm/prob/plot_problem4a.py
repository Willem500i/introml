import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([0, 1, 2, 3])
y_train = np.array([1, -1, 1, -1])
alpha = np.array([0, 0, 1, 1])
gamma = 3

x_plot = np.linspace(-1, 4, 500)
z = np.zeros_like(x_plot)

for i in range(len(x_train)):
    K = np.exp(-gamma * (x_train[i] - x_plot)**2)
    z += alpha[i] * y_train[i] * K

y_hat = np.sign(z)
y_hat[y_hat == 0] = 1

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(x_plot, z)
plt.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
plt.xlabel('x')
plt.ylabel('z')
plt.title('z vs x (gamma=3, alpha=[0,0,1,1])')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(x_plot, y_hat)
plt.scatter(x_train, y_train, color='red', s=100, zorder=5)
plt.xlabel('x')
plt.ylabel('y_hat')
plt.title('y_hat vs x (gamma=3, alpha=[0,0,1,1])')
plt.grid(True)
plt.tight_layout()
plt.savefig('problem4a_rbf.png', dpi=150, bbox_inches='tight')
plt.close()

