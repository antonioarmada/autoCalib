import pyglet
import numpy as np

# Crear el objeto Batch antes de la función update
batch = pyglet.graphics.Batch()

# Definir la función update y pasar el objeto Batch como argumento
def update(dt):

    #poligono = pyglet.shapes.Polygon([400, 100], [500, 10], [600, 100], [550, 175], [450, 150],color=(255, 255, 255, 255))

    coords = np.array( [[561 ,224],[641, 263],[601 ,347],[524, 307]],dtype=np.int32)
    coords =  coords.tolist()
    poligono = pyglet.shapes.Polygon(*coords,color=(255, 255, 255, 255))

    return poligono

# Programar la función update para que se llame repetidamente
pyglet.clock.schedule_interval(update, 1/30.0)

# Iniciar la ventana y el bucle de eventos
window = pyglet.window.Window(1440,1000)

@window.event
def on_draw():
    window.clear()
    # Llamar a la función update y obtener los elementos de dibujo devueltos
    poligono = update(0)
    poligono.draw()
    

pyglet.app.run()



