import pyglet

window = pyglet.window.Window()

# Cargar imagen desde archivo en disco
image = pyglet.image.load('background.jpg')



# Agregar sprite al lote de gráficos (batch) de la ventana
#batch = pyglet.graphics.Batch()

# Crear sprite con la imagen cargada y posición en la ventana
sprite = pyglet.sprite.Sprite(image, x=0, y=0)

@window.event
def on_draw():
    window.clear()
    #batch.draw()
    sprite.draw()

def reload_image():
    global sprite
    # Cargar nueva imagen desde archivo en disco
    new_image = pyglet.image.load('plantilla.jpg')
    # Actualizar sprite con nueva imagen
    sprite = pyglet.sprite.Sprite(new_image, x=0, y=0)
    # Redibujar ventana
    #batch.invalidate()

@window.event
def on_key_press(symbol, modifiers):
    if symbol == pyglet.window.key.R:
        print ("reload")
        reload_image()

pyglet.app.run()
