import numpy as np
import matplotlib.pyplot as plt

coordinates = np.array([(247.7, 29.62238482),(307, 40), (319.5, 39.61071404)])


x_t = np.gradient(coordinates[:, 0])
y_t = np.gradient(coordinates[:, 1])

vel = np.array([ [x_t[i], y_t[i]] for i in range(x_t.size)])

speed = np.sqrt(x_t * x_t + y_t * y_t)

tangent = np.array([1/speed] * 2).transpose() * vel

ss_t = np.gradient(speed)
xx_t = np.gradient(x_t)
yy_t = np.gradient(y_t)

curvature_val = np.abs(xx_t * y_t - x_t * yy_t) / (x_t * x_t + y_t * y_t)**1.5

print(curvature_val)