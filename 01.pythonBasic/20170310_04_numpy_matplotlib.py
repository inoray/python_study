import numpy as np
import matplotlib.pyplot as plt

print("- arrang 명령으로 -5부터 5까지 0.01 간격으로 구성된 배열 생성")
points = np.arange(-5, 5, 0.01)
print(points)

print("- meshgrid 명령으로 정방행렬 생성")
xs, ys = np.meshgrid(points, points)
print(xs)
print(ys)
print(xs.shape)

print("- 배열연산, np.sqrt(xs ** 2 + ys ** 2)")
z = np.sqrt(xs ** 2 + ys ** 2)
print(z)

# plot 출력
plt.imshow(z, cmap=plt.cm.gray)
plt.colorbar()
plt.title("Image plot of $\sqrt{x^2 + y^2}$ for a grid of values")
plt.show()
