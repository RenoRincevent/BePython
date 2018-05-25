import cv2
import time
import numpy as np
from threading import Thread,  Event

#Creation d'une variable globale qui s'incremente a chaque fois que l'on prend une nouvelle photo,
#Cela permet de na pas ecraser les anciennes photo prise lors de la capture video.
img_counter = 0

#takePicture() va enregistrer l'image en cours de la video et la stocker,
#puis dormir pendant n secondes, cela permet si il y a du mouvement permanent de ne pas prendre 
#toute les images mais une seule toute les n secondes (n etant fixe)
class takePicture(Thread):
    def __init__(self):
        Thread.__init__(self)
        self._stop_event = Event()

    def run(self):
        global img_counter
        imgName = ("frame_{}.png".format(img_counter))
        cv2.imwrite(imgName,frame)
        #print("{} ecrit".format(imgName))
        img_counter += 1
        time.sleep(1)

#Ouverture de la camera principale, ici celle de l'ordinateur
capture = cv2.VideoCapture(0)
#Ou ouverture de la camera distante, ex le RaspeberryPi
#capture = cv2.VideoCapture('/dev/stdin')

#Aucune frame n'a encore etait capture, prevFrame est donc initialise a None
prevFrame = None

#Creation du thread pour pouvoir prendre la photo par la suite
myThread = takePicture()

while True:
    (grabbed,frame) = capture.read()

    #Si la frame n'est pas lu correctement dans le buffer, on quitte la boucle
    if not grabbed:
        break

    #On passe l'image en niveau de gris et on lui applique un flou gaussien pour supprimmer le bruit
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(25,25), 0)

    if prevFrame is None:
        prevFrame = gray

    #On fait la difference absolue de l'image actuelle et la precedente
    #On fait un seuillage binaire sur cette nouvelle image
    #Puis on la dilate pour pouvoir plus facilement trouver les contours par la suite
    frameDelta = cv2.absdiff(prevFrame,gray)
    thresh = cv2.threshold(frameDelta, 7, 255, cv2.THRESH_BINARY)[1]
    kernel = np.ones((11,11),np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=2)

    #Recherche des contours des objets de l'image dilate
    (img,contr,hrchy) = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    #On initialise "area" a 0, cela va nous servir par la suite a voir si il y a un contour dans l'image
    #Et donc du mouvement
    area = 0

    for c in contr:

        #Tous les petits objets sont ignores avec cette ligne    
        if cv2.contourArea(c) < 1500:
			continue

        #On enregistre l'aire du contour pour la suite
        area = cv2.contourArea(c)

    #Si il y a une aire dans l'image, et donc un mouvement, et que le  thread n'est pas deja en train de
    #prendre une photo, alors on peut en relancer un autre pour prendre une photo. 
    if area != 0 and not(myThread.is_alive()):
        myThread = takePicture()
        myThread.start()

    #On affiche la video    
    cv2.imshow('Res',frame)
    #cv2.imshow('Threshold',thresh)

    #l'image actuelle devient la future image precedente
    prevFrame = gray

    #Quitte la capture video lorsque la touche q est appuye
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()