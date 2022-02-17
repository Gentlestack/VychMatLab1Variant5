# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import integrate

# 1
print("\n\t№1")
A = np.zeros((5, 5), dtype=float)
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        A[i][j] = np.random.uniform(-3, 3)
print(A)
A = A.transpose()
print(A)
print(np.linalg.det(A))

#2
print("\n\t№2")
vector = np.random.randint(-10, 10, (1, 3))
A = np.random.randint(-10, 10, (3, 2))
print(vector)
print(A)
multresult = np.dot(vector, A)
print(multresult)

#3
print("\n\t№3")
A = np.array([[-7, -5, -5], [0, 3, 0], [10, 5, 8]], int)
vals, vecs = np.linalg.eig(A)
print(vals)
print(vecs)

#4
print("\n\t№4")
print(integrate.quad(lambda x:(math.exp(2*x)*math.cos(x)), 0, math.pi/2))

#5
print("\n\t№5")
print(integrate.quad(lambda x:(1/(math.pow(x, 2) + 4*x + 9)), -math.inf, math.inf))

#6
# Создание объектов артборда и холста
print("\n\t№6")
plt.figure(figsize=(8, 5), dpi=80)
ax = plt.subplot(111)

# Мы решили удалить правую и верхнюю прямоугольные границы
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')


# Установить направление данных на координатной оси
 # 0 согласуется с нашей общей декартовой системой координат, 1 - противоположность
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data', 0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data', 0))


# Подготовить данные, использовать распаковку последовательности

X1 = np.linspace(0.0001, 3, 256, endpoint=False)
X2 = np.linspace(-3, 3, 256, endpoint= True)
C = np.log(X1) + 2
L = -3*X2

plt.plot(X1, C, color="blue", linewidth=2.5, linestyle="-", label="Log Function")
plt.plot(X2, L, color="red", linewidth=2.5, linestyle="-", label="Lin Function")

plt.xlim(X2.min() * 1.1, X2.max() * 1.1)

plt.legend(loc='upper left', frameon=False)
plt.grid()
plt.show()