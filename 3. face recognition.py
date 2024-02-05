import cv2
import numpy as np
import os 



recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')   #load trained model
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_SIMPLEX
#iniciate id counter, the number of persons you want to include
id = 2 #two persons (e.g. Jacob, Jack)


names = ['', 'Ryan Putranda K', 'Edwin Alexander']  #INPUT NAMA
prodi = ['','Master of Informatics Engineering', 'Informatics Science']  #INPUT NAMA
afiliasi = ['','Universitas Katolik Darma Cendika', 'UNIKA Darma Cendika']  #INPUT NAMA
nim = ['','06xxxx', '06xxxxx']  #INPUT NAMA

print(names[1])

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height

# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

while True:
    ret, img =cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )

    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,255), 2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        # Check if confidence is less them 100 ==> "0" is perfect match 
        
        kode = id
        confidence = "  {0}".format(round(100 - confidence))
        temp_confidence = int (confidence)
        if (temp_confidence > 50):
            id = names[kode]
            id_prodi = prodi[kode]
            id_afiliasi = afiliasi[kode]
            id_nim = nim[kode]
        else:
            id = "Tidak dikenal"
            id_prodi = "Tidak dikenal"
            id_afiliasi = "Tidak dikenal"
            id_nim = "Tidak dikenal"
        
        cv2.putText(img, "Researcher ID :", (x+200,y+20), font, 1, (255,255,255), 2)        
        cv2.putText(img, str(id), (x+200,y+40), font, 0.5, (255,255,255), 1)
        cv2.putText(img, str(id_prodi), (x+200,y+60), font, 0.5, (255,255,255), 1)
        cv2.putText(img, str(id_afiliasi), (x+200,y+80), font, 0.5, (255,255,255), 1)
        cv2.putText(img, str(id_nim), (x+198,y+100), font, 0.5, (255,255,255), 1)
        cv2.putText(img, "Prediction rate : "+str(confidence)+"%", (x+200,y+120), font, 0.5, (255,255,255), 1)
        
    
    cv2.imshow('PROJECT FACE RECOGNITION RYAN',img)

    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        print(str(id))
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
