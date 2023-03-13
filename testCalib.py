import numpy as np
import cv2, PIL
from cv2 import aruco
import json
import time
import pyglet

# mios:
from autoCalib import lee_json



# Función para actualizar la ventana con cada fotograma de video
def update(dt):
    """ probe usando una textura de sprite pero habia fuga de memoria
        frame_pg = pyglet.image.ImageData(frame.shape[1], frame.shape[0], 'BGR', frame.tobytes())
        frame_sprite = pyglet.sprite.Sprite(frame_pg)
        por lo que hice algo poco elegante pero funciona, grabo la captura en un archivo
        y la dibujo con .blit no como sprite. 
        El dibujo del poligono se podria hacer shaders se OpenGL pero no los entiendo bien"""
    
    global aviso

    # Genero un poligono cualqiera con opacidad 0 para que exista al dibujar si no entra el if
    poligono = pyglet.shapes.Polygon([400, 100], [500, 10], [600, 100], [550, 175], [450, 150],color=(255, 255, 255, 0))

    ret,frame = cap.read()

    if ret:
        cv2.imwrite('captura.jpg',frame)
        esquinas = busca_marcador(frame)
        texto_aviso = "NO SE DETECTARON MARCADORES"
        
        if esquinas.size > 0:       #uso .size porque es un array de np
            
            esquinas = esquinas.astype(np.float32)
            # Agregamos una dimensión extra para poder utilizar perspectiveTransform
            esquinas= np.expand_dims(esquinas, axis=1)
            esquinas_corregidas = cv2.perspectiveTransform(esquinas, matriz_array)
            # Quitamos la dimensión extra agregada anteriormente
            esquinas_corregidas = np.squeeze(esquinas_corregidas, axis=1)
            esquinas_ls = esquinas_corregidas.tolist()
            poligono = pyglet.shapes.Polygon(*esquinas_ls,color=(255, 255, 255, 255)) # * desempaca la lista

        aviso = pyglet.text.Label(texto_aviso,
                          font_name='Arial',
                          font_size=14,
                          x= win_captura.width//2, y=win_captura.height-30,
                          anchor_x='center', anchor_y='center')
    return poligono
        
# Crear una función que maneje el evento de teclado
def on_key_press(symbol, modifiers):
    if symbol == pyglet.window.key.ESCAPE:
        pyglet.app.exit()



def busca_marcador(imagen):
    """
    Busca 1 marcadores en la imagen pasada como argumento,
    si se encuentra  devuelve una lista con las coordenadas de
    las esquinas-
    Sino se encuentran devuelve una lista vacía como coordenadas
    """
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50) 
    parametros = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parametros)
    esquinas, ids, rechazados = detector.detectMarkers(imagen)
    esquinas_resultado = np.array([]).astype(int)

    # si ya vio un marcador
    if len(esquinas) > 0: #ids is not None:
        esquinas_resultado= np.array(esquinas).astype(int)
        #print (ids)
        # me quedaba con una dimesion de longitud uno que se la saco
        esquinas_resultado = esquinas_resultado.squeeze()
        # extraigo la primera esquina (sup izq) de cada marcador
    
    # Devolviendo estaba la ganza

    return esquinas_resultado





# ---------  Main  ---------------------------------------

if __name__ == '__main__':


    # lee la config de JSON -
    res_proyector_w, res_proyector_h, resolucion_camara_w, resolucion_camara_h, id_camara, \
        ancho_marcador, separacion_al_borde, matriz = lee_json('configs.json')
    
    print ("--------")
    print (matriz)
    print ("--------")

    matriz_array = np.array(matriz, dtype=np.float32)
    matriz_array = matriz_array.reshape(3, 3)


    print ("--------")
    print (matriz_array)
    print ("--------")


    # Obtener una lista de todas las pantallas disponibles
    screens = pyglet.canvas.Display().get_screens()
    print (screens)

    # Si no se especificaron valores para estas variables en configs.json, los genera 
    if res_proyector_w == 0 or res_proyector_h == 0 :
        res_proyector_w = screens[1].width
        res_proyector_h = screens[1].height
        print (res_proyector_w)
    if ancho_marcador == 0:
        ancho_marcador = int(res_proyector_w * .1)
        print (ancho_marcador)
    if separacion_al_borde == 0:
        separacion_al_borde = int(ancho_marcador*.2)

    # Configurar las ventanas de Pyglet las dos pantallas
    win_captura = pyglet.window.Window(resolucion_camara_w, resolucion_camara_h,
                                    screen=screens[0], caption='Captura RAW')
    win_proyector = pyglet.window.Window(screens[1].width, screens[1].height,
                                fullscreen=False, # cambiar a TRUE en implementacion
                                resizable=True, 
                                screen=screens[1],
                                caption='Plantilla')

    # Configurar la posición de la ventana en la pantalla secundaria
    win_proyector.set_location(screens[1].x,screens[1].y)



    # Abre camara
    try:
        cap = cv2.VideoCapture(id_camara)  # lo saca del JSON 0 si hay una camara
        cap.set(3,resolucion_camara_w)
        cap.set(4,resolucion_camara_h)
        time.sleep(2)
        print("-- CAMARA ENCONTRADA --- ")
    except Exception as e:
        print("-- NO SE ENCUENTRA LA CÁMARA -------")
        print(str(e))
        #sys.exit()
        quit()

    # Cargar la captura inicial
    ret, frame = cap.read()
    if ret:
        # ver comentario en funcion 'update'
        cv2.imwrite('captura.jpg',frame)

    # Configurar el evento de actualización de la ventana de la cámara
    a = pyglet.clock.schedule_interval(update, 1/30.0)
    print (a)

    # defino el rectangulo que voy a actualizar luego con la pos del marcador. 
    batch_marca = pyglet.graphics.Batch()

    line2 = pyglet.shapes.Line(150, 150, 444, 111, width=4, color=(200, 20, 20), batch=batch_marca)

    # Pongo un fondo en la ventana del proyector
    fondo = pyglet.image.load('background.jpg')
    sprite_fondo = pyglet.sprite.Sprite(fondo)
    sprite_fondo.scale = max(res_proyector_w / sprite_fondo.width, res_proyector_h / sprite_fondo.height)
    sprite_fondo.position = (res_proyector_w - sprite_fondo.width) / 2, (res_proyector_h - sprite_fondo.height) / 2 , 0 


    # creo los manejadores de evento de teclado
    # no uso '.on_key_press' porque es para una sola ventana
    win_captura.push_handlers(on_key_press)
    win_proyector.push_handlers(on_key_press)

    # genero la etiqueta inicial en la ventana de la captura
    aviso = pyglet.text.Label('NO SE ENCUENTRA MARCADOR ArUCO',
                          font_name='Arial',
                          font_size=14,
                          x= win_captura.width//2, y=win_captura.height-30,
                          anchor_x='center', anchor_y='center')

    # Configurar el evento de dibujado de la ventana con la captura
    @win_captura.event
    def on_draw():
        win_captura.clear()
        #frame_sprite.draw()
        image = pyglet.image.load('captura.jpg')
        image.blit(0,0)
        aviso.draw()
        
    win_proyector.flip()

    # Configurar el evento de dibujado del proyector
    @win_proyector.event
    def on_draw():
        win_proyector.clear()
        sprite_fondo.draw()
        linea1 = update(0)
        linea1.draw()


    # Iniciar la aplicación de Pyglet
    pyglet.app.run()

    # Liberar la captura de video al finalizar
    cap.release()