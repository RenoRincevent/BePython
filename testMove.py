import cv2
import numpy as np

#Ouverture de la camera principale, ici celle de l'ordinateur
capture = cv2.VideoCapture(0)
#Ou ouverture de la camera distante, ex le RaspeberryPi
#capture = cv2.VideoCapture('/dev/stdin')

#Aucune frame n'a encore etait capture, prevFrame est donc initialise a None
prevFrame = None


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

    #Ce mask va nous servir a encadrer l'objet de la couleur de celui ci
    mask = np.zeros(frame.shape[:2],np.uint8)

    for c in contr:

        #Tous les petits objets sont ignores avec cette ligne
        if cv2.contourArea(c) < 1500:
			continue

        #Recherche des coordonnees de l'objet et on adapte le mask en fonction de l'objet
        (x,y,w,h) = cv2.boundingRect(c) 
        mask[y:y+h, x:x+w] = 255

        #Recuperation de la couleur du pixel du milieu
        #On suppose qu'on est dans le cas ou l'objet a une couleur uniforme
        (bl,gr,re) = frame[y+(h/2),x+(w/2)]
        bl = int(bl) 
        gr = int(gr)
        re = int(re)
        couleur ='B:'+ str(bl) + 'G:' + str(gr) + 'R:' + str(re)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame,couleur,(x+w,y), font, 1, (255,255,255), 2, cv2.LINE_AA)

        #Le rectangle prend la couleur de l'objet et les nuances de bleu,vert,rouge sont affiche sur le cote
        cv2.rectangle(frame,(x,y),(x+w,y+h),(bl,gr,re),3)
        
    masked_img = cv2.bitwise_and(frame,frame,mask=mask)    
    
    #On affiche la video avec les rectangles
    cv2.imshow('contour',frame)

    #Les autres video (seuillage,masque...) pour tests
    #cv2.imshow('blur',gray)
    #cv2.imshow('res',frameDelta)
    #cv2.imshow('mask',masked_img)
    #cv2.imshow('res',thresh)

    #l'image actuelle devient la future image precedente
    prevFrame = gray

    #Quitte la capture video lorsque la touche q est appuye
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()