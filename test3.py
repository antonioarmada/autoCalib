import cv2
import numpy as np

# Coordenadas de entrada
coords = np.array([[624, 97], [685, 139], [647, 213], [588, 171]], dtype=np.float32)

# Matriz de transformación homográfica
H = np.array([[1.50329096e+00, 4.56075843e-02, -2.62856187e+02],
              [1.66241164e-02, 1.51675008e+00, -1.08363358e+02],
              [1.82582119e-05, 7.63838947e-05, 1.00000000e+00]])

# Agregamos una dimensión extra para poder utilizar perspectiveTransform
coords = np.expand_dims(coords, axis=1)

# Transformamos las coordenadas utilizando perspectiveTransform
transformed_coords = cv2.perspectiveTransform(coords, H)

# Quitamos la dimensión extra agregada anteriormente
transformed_coords = np.squeeze(transformed_coords, axis=1)

print(transformed_coords)
