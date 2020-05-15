#!/usr/bin/env python
import cv2
import requests

face_detection = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while face_detection.empty():
    url = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'
    downloded = requests.get(url)
    open('haarcascade_frontalface_default.xml','wb').write(downloded.content)
    face_detection = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")   

def detect_face(img):
    face_img = img.copy()
    
    face_rectangle = face_detection.detectMultiScale(face_img)
    
    for (x,y,w,h) in face_rectangle:
        cv2.rectangle(face_img,
                     (x,y),
                     (x+w,y+h),
                     color=(255,255,255),
                     thickness=10)
    return face_img

cap = cv2.VideoCapture(0)

if cap.isOpened():
    while True:
        ret,frame = cap.read()

        cv2.imshow("Face Detection",detect_face(frame))

        if cv2.waitKey(15)& 0xFF==27:
            break
    cap.release()
    cv2.destroyAllWindows()
else:
    print("Error in opening resource")

    

