import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 2*np.pi, 1000)
x = 16*np.sin(t)**3
y = 13*np.cos(t) - 5*np.cos(2*t) - 2*np.cos(3*t) - np.cos(4*t)

plt.figure(figsize=(8, 6))
plt.plot(x, y, 'red', linewidth=2)
plt.title('Heart Function')
plt.axis('equal')
plt.grid(True)
plt.show()
