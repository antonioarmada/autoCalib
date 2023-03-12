import numpy as np
import cv2, PIL
from cv2 import aruco
import time
import datetime
import json
import pyglet

"""
Genera una plantilla con 4 marcadores a 10px de las esquinas
Inicia captura de webcam y espera que se apriete 'd' para detectar marcadores
si detecta marcadores, genera matriz homografica y muestra caputra corregida
--
Funciona con OpenCV 4.7 no como todos los ejemplos que andan dando vuelta


#sacar del Json cuando esto funcione -------------------
    res_proyector_w = screens[1].width
    res_proyector_h = screens[1].height

calcular el ancho y separacion de los marcadores en relacion tamaño proyeccion
    
"""
# --------- Funciones de Interfaz  ------------------------------


# Función para actualizar la ventana con cada fotograma de video
def update(dt):
    """ probe usando una textura de sprite pero habia fuga de memoria
        frame_pg = pyglet.image.ImageData(frame.shape[1], frame.shape[0], 'BGR', frame.tobytes())
        frame_sprite = pyglet.sprite.Sprite(frame_pg)
        por lo que hice algo poco elegante pero funciona, grabo la captura en un archivo
        y la dibujo con .blit no como sprite. """
    ret,frame = cap.read()
    #frame = cv2.flip(frame,0)
    cv2.imwrite('captura.jpg',frame)



# Crear una función que maneje el evento de teclado
def on_key_press(symbol, modifiers):

    global aviso # para actualizar la etiqueta en la ventana de captura

    if symbol == pyglet.window.key.ESCAPE:
        pyglet.app.exit()

    key = pyglet.window.key.symbol_string(symbol).lower() #por si es D
    
    # al presionar d se intenta la detección. 
    if key == "d":
        print("La tecla D ha sido presionada")
        frame = cv2.imread('captura.jpg')
        imagen, coord_detectados = detecta_marcadores(frame)
        print(coord_detectados)
        # Si devolvio las coord de los 4 marcadores
        if not coord_detectados == []:
            print ("Coord de los marcadores dectados:")
            se_detectaron_marcadores(imagen,coord_detectados)
        else:
            print ('no se detectaron los 4 marcadores')
            aviso = pyglet.text.Label(f'NO SE DETECTARON LOS 4 MARCADORES ({time.time()})',
                          font_name='Arial',
                          font_size=20,
                          x=win_corregida.width//2, y=win_corregida.height//2,
                          anchor_x='center', anchor_y='center')
            

# Cuando se detectaron los marcadores         
def se_detectaron_marcadores(imagen, coord_detectados):

    global sprite_captura_corregida, aviso
    
    print ("Coord de los marcadores originales:")
    print (coord_marcadores)
   
    print ("Matriz de transformación homográfica:")
    mtx = get_homography_matrix(coord_detectados, coord_marcadores)
    print (mtx)
    escribe_json ('configs.json', mtx)

    # Generlo la imagen corregida
    captura_corregida = cv2.warpPerspective(imagen, mtx, (res_proyector_w, res_proyector_h))
    cv2.imwrite ("corregido.jpg", captura_corregida)
    new_image = pyglet.image.load('corregido.jpg')
    sprite_captura_corregida = pyglet.sprite.Sprite(new_image, x=0, y=0)
    # borro el aviso
    aviso = pyglet.text.Label("",
                          font_name='Arial',
                          font_size=20,
                          x=win_corregida.width//2, y=win_corregida.height//2,
                          anchor_x='center', anchor_y='center')
    # Redibujar ventana
    #win_corregida.invalidatae()






# --------- Funciones de Planatilla y transformación ------------

def lee_json(ruta):
    """
    Lee el archivo de configuración y devuelve las variables que
    hay adentro

    Arg.
    Ruta del archivo de configuración

    Return.
    Varialbes de configuración
    """
    with open(ruta, 'r') as f:
        configs = json.load(f)
    # accede a los datos en el diccionario generado
    res_proyector_w = configs["dispositivos"]["forzar_res_proyector_w"]
    res_proyector_h = configs["dispositivos"]["forzar_res_proyector_h"]
    resolucion_camara_w = configs["dispositivos"]["resolucion_camara_w"]
    resolucion_camara_h = configs["dispositivos"]["resolucion_camara_h"]
    id_camara = configs["dispositivos"]["id_camara"]
    ancho_marcador = configs["marcadores"]["forzar_ancho_marcador"]
    separacion_al_borde = configs["marcadores"]["forzar_separacion_al_borde"]
    matriz= configs['resultados']['matriz_transformacion']
    return (res_proyector_w,res_proyector_h, \
            resolucion_camara_w,resolucion_camara_h, id_camara, \
            ancho_marcador,separacion_al_borde,matriz)

def escribe_json(ruta , matriz):
    """
    Actualiza la variable de matriz de transformación en el archivo Json
    para que no tenga que quede guardada la calibración y no tenga que repetirse
    
    IMPORTANTE: serializo la matriz, array de numpy, para poder guardarla en el JSON,
    para usarla posiblemente tenga que volverla a convertir a un array de numy
    """

    with open(ruta, 'r') as f: #r de read
        data = json.load(f)
    
    matriz = matriz.tolist() #np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    data['resultados']['matriz_transformacion'] = matriz

    with open('configs.json', 'w') as f: #w de write ;)
        json.dump(data, f, indent=4)
    
    print (ruta + " ACTUALIZADO")




def genera_plantilla(res_proyector_w=800,res_proyector_h=600,separacion_al_borde=10,ancho_marcador=100): # type: (int, int, int, int) -> Tuple[np.ndarray, List[Tuple[int, int]]]
    """
    Genera una plantilla con 4 marcadores ARuco 4x4_50 a una distancia dada de las esquinas. 
    Devuelve la imagen con la plantilla y una lista con las coordenadas de centros de marcadores.
    """
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50) 

    # creo una imagen para el fondo blanco
    plantilla = np.zeros((res_proyector_h, res_proyector_w), np.uint8)
    plantilla.fill(255)

    # Empiezo a colocar los 4 marcadores
    # Guardo las coord de la esquina sup izq de cada uno
    coord_marcadores = []

    # 1 arriba izq
    marcador = aruco.generateImageMarker(aruco_dict,1, ancho_marcador)
    x, y = separacion_al_borde, separacion_al_borde
    plantilla[y:y+marcador.shape[0], x:x+marcador.shape[1]] = marcador #y:y.img.shap[0] define una rea de interes RoI en los pixels verticales que va de y al alto de la imagen
    coord_marcadores.append((x,y))

    # 2  arriba der
    marcador = aruco.generateImageMarker(aruco_dict,2, ancho_marcador)
    x = res_proyector_w-separacion_al_borde-marcador.shape[1]
    plantilla[y:y+marcador.shape[0], x:x+marcador.shape[1]] = marcador
    coord_marcadores.append((x,y))

    # 3  abajo der
    marcador = aruco.generateImageMarker(aruco_dict,3, ancho_marcador)
    y = res_proyector_h - separacion_al_borde - marcador.shape[0]
    plantilla[y:y+marcador.shape[0], x:x+marcador.shape[1]] = marcador
    coord_marcadores.append((x,y))
    # 4  abajo izq

    marcador = aruco.generateImageMarker(aruco_dict,4, ancho_marcador)
    x, y = separacion_al_borde,  res_proyector_h - separacion_al_borde - marcador.shape[0]
    plantilla[y:y+marcador.shape[0], x:x+marcador.shape[1]] = marcador
    coord_marcadores.append((x,y))

    # Convertir la imagen a RGB para que funcine bien en PyGlet y otros
    plantilla_rgb = cv2.cvtColor(plantilla, cv2.COLOR_GRAY2RGB)

    font = cv2.FONT_HERSHEY_SIMPLEX
    x , y = int(plantilla_rgb.shape[1]/2) -350 , int(plantilla_rgb.shape[0]/2)
    cv2.putText(plantilla_rgb,"ESTE TEXTO SE DEBE VER AL DERECHO EN LA PROYECCION Y EN LA CAPTURA",
                (x,y), font, .5,(0,0,0),1,cv2.LINE_AA)

    return plantilla_rgb, coord_marcadores




def detecta_marcadores(imagen):
    """
    Busca 4 marcadores en la imagen pasada como argumento,
    si se encuentran 4 devuelve la imagen con el polígono que une sus
    primeras esqinas (arriba izq) y una lista con las coordenadas de estas
    Sino se encuentran marcadores imprime "FALTAN MARCADORES"
    y devuelve una lista vacía como coordenadas
    """
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50) 
    parametros = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parametros)
    esquinas, ids, rechazados = detector.detectMarkers(imagen)
    primeras_esquinas = []

    # si ya vio los 4 marcadores
    if ids is not None and len(ids) == 4 :
        # Obtener los índices ordenados por columna
        ids_ordenados = np.argsort(ids, axis=0)
        esquinas = np.array(esquinas).astype(int)
        # Aplicar el mismo orden a los dos arrays
        ids_ordenado = ids[ids_ordenados].flatten()
        esquinas_ordenado = esquinas[ids_ordenados]
        # me quedaba con una dimesion de longitud uno que se la saco
        esquinas_ordenado = esquinas_ordenado.squeeze()
        # extraigo la primera esquina (sup izq) de cada marcador

        for i in range(0, len(esquinas_ordenado), 1):
            primeras_esquinas.append((int(esquinas_ordenado[i][0][0]),int(esquinas_ordenado[i][0][1])))
        # si la imagen es en escala de grises la convierte a color BGR
        if imagen.shape [2] == 1: # cantidad de canales
            imagen = cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)
        # Dibujo los centros de las esquinas en la imagen
        for esquina in primeras_esquinas:
            cv2.circle(imagen, esquina, 10, (0, 0, 255), 1)

        # Imprimo un rectangulo uniendolas
        puntos = np.array(primeras_esquinas, np.int32)
        puntos = puntos.reshape((-1, 1, 2))
        cv2.polylines(imagen,[puntos], True, (0, 255, 0), thickness=1)

    else :
        # si no hay 4 marcadores imprimo cartel
        font = cv2.FONT_HERSHEY_SIMPLEX
        x , y = int(imagen.shape[1]/2) -120 , int(imagen.shape[0]/2)
        cv2.putText(imagen,"FALTAN MARCADORES",(x,y), font, .5,(0,0,255),1,cv2.LINE_AA)
    
    # Devolviendo estaba la ganza
    return imagen, primeras_esquinas



def get_homography_matrix(source, destination):
    """ Calculates the entries of the Homography matrix between two sets of matching points.

    Args
    ----
        - `source`: Source points where each point is int (x, y) format.
        - `destination`: Destination points where each point is int (x, y) format.

    Returns
    ----
        - A numpy array of shape (3, 3) representing the Homography matrix.

    Raises
    ----
        - `source` and `destination` is lew than four points.
        - `source` and `destination` is of different size.
    """
    assert len(source) >= 4, "must provide more than 4 source points"
    assert len(destination) >= 4, "must provide more than 4 destination points"
    assert len(source) == len(destination), "source and destination must be of equal length"
    A = []
    b = []
    for i in range(len(source)):
        s_x, s_y = source[i]
        d_x, d_y = destination[i]
        A.append([s_x, s_y, 1, 0, 0, 0, (-d_x)*(s_x), (-d_x)*(s_y)])
        A.append([0, 0, 0, s_x, s_y, 1, (-d_y)*(s_x), (-d_y)*(s_y)])
        b += [d_x, d_y]
    A = np.array(A)
    h = np.linalg.lstsq(A, b,rcond=None)[0]
    h = np.concatenate((h, [1]), axis=-1)
    h = np.reshape(h, (3, 3))
    return h





# ---------  Main  ---------------------------------------

if __name__ == '__main__':

    # lee la config de JSON -
    # si bien la funcion devuelve la matriz guardada, en este
    # código busco re-generarla. Está asi para utilizar la misma
    # función en otro código
    res_proyector_w, res_proyector_h, resolucion_camara_w, resolucion_camara_h, id_camara, \
        ancho_marcador, separacion_al_borde, matriz = lee_json('configs.json')



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
    win_corregida = pyglet.window.Window(res_proyector_w, res_proyector_h,
                                   screen=screens[0], caption='Captura Corregida')
    win_proyector = pyglet.window.Window(screens[1].width, screens[1].height,
                               fullscreen=False, # cambiar a TRUE en implementacion
                               resizable=True, 
                               screen=screens[1],
                               caption='Plantilla')
    
    # Configurar la posición de la ventana en la pantalla secundaria
    win_proyector.set_location(screens[1].x,screens[1].y)

    # Configurar la posición de las ventanas del monitor
    win_captura.set_location(screens[0].x,screens[0].y)
    win_corregida.set_location(screens[0].width-resolucion_camara_w,screens[0].y)


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


    # Genero la plantilla y la convierto a imagen de PyGlet
    plantilla, coord_marcadores = genera_plantilla(res_proyector_w, res_proyector_h,
                                                   separacion_al_borde, ancho_marcador)
    # Genero la imagen y su sprite de Pyglet
    plantilla_pg = pyglet.image.ImageData(plantilla.shape[1], plantilla.shape[0], 'RGB', plantilla.tobytes())
    plantilla_sprite = pyglet.sprite.Sprite(plantilla_pg)
    #Aca espejo la plantilla, no se si es un problema de macOS
    plantilla_sprite.scale_y = -1
    # al espejar el sprite cambia el punto de anclaje y se sale de la ventana, lo soluciono asi:
    plantilla_sprite.y = plantilla_sprite.height

    # genero la etiqueta inicial en la ventana Corregida
    aviso = pyglet.text.Label('PRESIONAR "d" PARA INICIAR LA DETECCIÓN',
                          font_name='Arial',
                          font_size=25,
                          x=win_corregida.width//2, y=win_corregida.height//2,
                          anchor_x='center', anchor_y='center')


    # Cargar la captura inicial
    ret, frame = cap.read()
    if ret:
        # ver comentario en funcion 'update'
        cv2.imwrite('captura.jpg',frame)

    # Configurar el evento de actualización de la ventana de la cámara
    pyglet.clock.schedule_interval(update, 1/30.0)
    
    # creo los manejadores de evento de teclado
    # no uso '.on_key_press' porque es para una sola ventana
    win_captura.push_handlers(on_key_press)
    win_proyector.push_handlers(on_key_press)
    win_corregida.push_handlers(on_key_press)
    
    # Pongo un fondo en la ventana de la captura para despues poder actualizarlo.
    fondo = pyglet.image.load('background.jpg')
    sprite_captura_corregida = pyglet.sprite.Sprite(fondo)
    sprite_captura_corregida.scale = max(res_proyector_w / sprite_captura_corregida.width, res_proyector_h / sprite_captura_corregida.height)
    sprite_captura_corregida.position = (res_proyector_w - sprite_captura_corregida.width) / 2, (res_proyector_h - sprite_captura_corregida.height) / 2 , 0 


    # Configurar el evento de dibujado de la ventana con la captura
    @win_captura.event
    def on_draw():
        win_captura.clear()
        #frame_sprite.draw()
        image = pyglet.image.load('captura.jpg')
        image.blit(0,0)

    # Configurar el evento de dibujado de la ventana con la correccion
    @win_corregida.event
    def on_draw():
        win_corregida.clear()
        sprite_captura_corregida.draw()
        aviso.draw()

    # Configurar el evento de dibujado del proyector
    @win_proyector.event
    def on_draw():
        win_proyector.clear()
        plantilla_sprite.draw()


    # Iniciar la aplicación de Pyglet
    pyglet.app.run()

    # Liberar la captura de video al finalizar
    cap.release()


"""
# Cargar la imagen original
imagen_original = pyglet.image.load('imagen.jpg')

# Crear una ventana y un sprite de la imagen
ventana = pyglet.window.Window()
sprite = pyglet.sprite.Sprite(imagen_original)

# Dibujar el sprite en la ventana
@ventana.event
def on_draw():
    ventana.clear()
    sprite.draw()

# Función para actualizar la imagen de la ventana
def actualizar_imagen():
    # Cargar la nueva imagen
    nueva_imagen = pyglet.image.load('nueva_imagen.jpg')
    # Actualizar el sprite con la nueva imagen
    sprite.image = nueva_imagen
    # Actualizar la imagen de la ventana con el nuevo sprite
    ventana.set_image(sprite.image.get_texture())

# Ejecutar la función de actualización al presionar la tecla "U"
@ventana.event
def on_key_press(symbol, modifiers):
    if symbol == pyglet.window.key.U:
        actualizar_imagen()

pyglet.app.run()



import numpy as np
import cv2

# Punto de entrada
punto_entrada = np.array([[[x, y]]], dtype=np.float32)

# Transformación homográfica
mtx = np.array([[a, b, c], [d, e, f], [g, h, 1]])

# Transformar el punto de entrada
punto_salida = cv2.perspectiveTransform(punto_entrada, mtx)

# Obtener las coordenadas del punto de salida
X, Y = punto_salida[0][0]

# Imprimir las coordenadas del punto de salida
print(f"El punto ({x}, {y}) transformado por la matriz de transformación homográfica es: ({X}, {Y})")
"""

