# 출처: http://pinkwink.kr/571

import numpy as np
import matplotlib.pyplot as plt

a = np.array([[1, 2], [3, 4]])
b = np.array([[1, 0], [1, 1]])

print(a)
print(b)

c = a * b
print(c)

d = np.dot(a, b)
print(d)

t = np.linspace(0, 2*np.pi, 100)
y1 = np.sin(t)
y2 = np.cos(t)
plt.plot(t, y1, t, y2)
plt.show()
