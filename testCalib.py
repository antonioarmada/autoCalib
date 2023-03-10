import numpy as np
import cv2, PIL
from cv2 import aruco
import json
import time
import pyglet

# mios:
from autoCalib import lee_json


# Abre camara
try:
    cap = cv2.VideoCapture(1)  # 0 es el ID de la cámara predeterminada
    cap.set(3,resulucion_camara_w)
    cap.set(4,resulucion_camara_h)
    time.sleep(2)
    print("-- CAMARA ENCONTRADA --- ")
except Exception as e:
    print("-- NO SE ENCUENTRA LA CÁMARA -------")
    print(str(e))
    #sys.exit()
    quit()

# Obtener una lista de todas las pantallas disponibles
screens = pyglet.canvas.Display().get_screens()
print (screens)

# Configurar las ventanas de Pyglet las dos pantallas
window0 = pyglet.window.Window(screen=screens[0])
window1 = pyglet.window.Window(screens[1].width, screens[1].height,
                               fullscreen=False, # cambiar a TRUE en implementacion
                               resizable=True, 
                               screen=screens[1])
# Configurar la posición de la ventana en la pantalla secundaria
window1.set_location(screens[1].x,screens[1].y)

# Función para actualizar la ventana con cada fotograma de video
def update(dt):
    ret, frame = cap.read()
    if ret:
        image = pyglet.image.ImageData(frame.shape[1], frame.shape[0], 'BGR', frame.tobytes())
        sprite.image = image

# Cargar la imagen inicial
ret, frame = cap.read()
if ret:
    image = pyglet.image.ImageData(frame.shape[1], frame.shape[0], 'BGR', frame.tobytes())
    sprite = pyglet.sprite.Sprite(image)

# Configurar el evento de actualización de la ventana
pyglet.clock.schedule_interval(update, 1/30.0)


window1.switch_to()
# Crear un objeto Batch
batch1 = pyglet.graphics.Batch()
# Agregar un rectángulo al Batch
rectangle = pyglet.shapes.Rectangle(x=200, y=200, width=100, height=50, color=(0, 255, 0), batch=batch1)

# Configurar el evento de dibujado de la ventana
@window0.event
def on_draw():
    window0.clear()
    sprite.draw()

@window1.event
def on_draw():
    # Establecer el color de fondo a rojo
    window1.clear()
    batch1.draw()
    

# Iniciar la aplicación de Pyglet
pyglet.app.run()

# Liberar la captura de video al finalizar
cap.release()
