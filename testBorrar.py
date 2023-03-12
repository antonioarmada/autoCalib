import pyglet
import cv2
window = pyglet.window.Window()

video = cv2.VideoCapture(0)
def takepicture(dt):
    num = 0
    ret,frame = video.read()
    cv2.imwrite(str(num)+'.jpg',frame)
    #print("Image_Captured")

@window.event
def on_draw():
    window.clear()
    image = pyglet.image.load('0.jpg')
    image.blit(0,0)

pyglet.clock.schedule_interval(takepicture, 0.001)

pyglet.app.run()