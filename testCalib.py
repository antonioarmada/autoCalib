import numpy as np
import cv2, PIL
from cv2 import aruco
import json
import time
# mios:
from autoCalib import lee_json

if __name__ == '__main__':

    # lee la config de JSON -
    res_proyector_w, res_proyector_h, resulucion_camara_w, resulucion_camara_h, \
        ancho_marcador, separacion_al_borde, matriz = lee_json('configs.json')
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

    ret, frame = cap.read()
    
    # mientras tenga fotoramas nuevos
    while ret:
        ret, frame = cap.read()
        cv2.imshow("Captura",frame)

        #convierto la matriz que guarde en lista en JSON (no se puede guardar np array)
        matriz = np.array(matriz)
        
        # transformo y muestro
        destination_image = cv2.warpPerspective(frame, matriz, (res_proyector_w, res_proyector_h))
        cv2.imshow("Corregido",destination_image)
        
        #espera q para salir
        if cv2.waitKey(1) & 255 == ord('q'):
            quit()
    
    cap.release()
    cv2.destroyAllWindows()