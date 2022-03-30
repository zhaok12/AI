import cv2
import math
import numpy as np

# Enable camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 1920)
cap.set(4, 1080)

# import cascade file for facial recognition
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
#faceCascade = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")

'''
    # if you want to detect any object for example eyes, use one more layer of classifier as below:
    eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")
'''
eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")
#eyeCascade = cv2.CascadeClassifier("data/haarcascade_eye_tree_eyeglasses.xml")

while True:
    success, img = cap.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Getting corners around the face
    faces = faceCascade.detectMultiScale(imgGray, 1.3, 5)  # 1.3 = scale factor, 5 = minimum neighbor
    # drawing bounding box around face
    for (x, y, w, h) in faces:
        img2 = cv2.putText(img, "face", (x,y-int(w/32)), cv2.FONT_HERSHEY_PLAIN, 4.0, (255,0,255), 2,)
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # detecting eyes
    eyes = eyeCascade.detectMultiScale(imgGray, 1.3, 5)
    # drawing bounding box for eyes
    for (ex, ey, ew, eh) in eyes:
        #img = cv2.circle(img, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 3)
        #putText(img, label, Point(x, y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
        img2 = cv2.putText(img, "eye", (ex+int(ew/4),ey+int(ew/4)), cv2.FONT_HERSHEY_PLAIN, 1.0, (0,255,255), 2,)
        img = cv2.circle(img, (int(ex+ew/2), int(ey+eh/2)), (int(math.sqrt((ew*ew)/64+(ey*ey)/64))), (255, 0, 0), 3)
    
    cv2.imshow('face_detect', img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyWindow('face_detect')
