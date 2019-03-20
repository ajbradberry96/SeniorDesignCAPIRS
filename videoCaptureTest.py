##this comes from
##https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html

import tensorflow as tf
from PIL import Image

import numpy as np
import forward_model
import cv2
import threading
from apscheduler.schedulers.background import BackgroundScheduler
import forward_model
import face_recognition

Labels = forward_model.get_imagenet_labels()
##Using the frame as a global variable because problems passing
##parameters with scheduler
frame = np.empty(1)

##set up for face stuff
knownEncodings = []
ozImage = face_recognition.load_image_file("media/Oz1.png")
bgImage = face_recognition.load_image_file("media/BG1.png")
ozEncoding = face_recognition.face_encodings(ozImage)
bgEncoding = face_recognition.face_encodings(bgImage)
knownEncodings.append(ozEncoding)
knownEncodings.append(bgEncoding)
#need to find a way to get both encodings to work at the same time
print("Known encodings are: ")
print(knownEncodings)

def printClasses():
    """
    Take in current Frame and classify it
    giving the likely probability
    """
    img = Image.fromarray(frame)
    probs = forward_model.predict(img)
    topk = list(probs.argsort()[-10:][::-1])
    topprobs = probs[topk]
    imgClass = [Labels[i][:15] for i in topk]
    print(topprobs[1])
    print(imgClass[1])

def faceChecker():
    """
    Take an image and check with list of known faces
    print off results for simplicity
    """
    #get encoding for image
    if(len(face_recognition.face_encodings(frame)) > 0):
        unknownEncoding = face_recognition.face_encodings(frame)[0]
        unknownImg = Image.fromarray(frame)
        #check for josh
        print(face_recognition.compare_faces(ozEncoding, unknownEncoding))
        #check for brian
        print(face_recognition.compare_faces(bgEncoding, unknownEncoding))
        print("NEXT")
    else:
        print("no face here") 
    
#create tensorflow session
tf.logging.set_verbosity(tf.logging.ERROR)
sess = tf.InteractiveSession()
#initialize stuff
forward_model.init(sess)

#create video capture object, 0 is the first camera
cap = cv2.VideoCapture(0)

#some initial conditions
first = True
sched = BackgroundScheduler()
sched.start()

#loop to get video
while(True):
    #capture frame by frame
    ret, frame = cap.read()

    #classify frame every second
    if(first):
        #sched.add_job(printClasses, 'interval', seconds = 10)
        sched.add_job(faceChecker, 'interval', seconds = 5)
        first = False
    
    #display the resulting frame
    cv2.imshow('frame', frame)
    #hit q while active in picture to quit program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#when done release everything
cap.release()
cv2.destroyAllWindows()
sched.shutdown()
