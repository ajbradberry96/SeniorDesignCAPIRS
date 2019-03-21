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

##set up known faces
knownEncodings = []
names = []
#set up for Josh
ozImage = face_recognition.load_image_file("media/Oz1.png")
ozEncoding = face_recognition.face_encodings(ozImage)
knownEncodings.append(ozEncoding)
names.append("Josh")
#set up for Brian
bgImage = face_recognition.load_image_file("media/BG1.png")
bgEncoding = face_recognition.face_encodings(bgImage)
knownEncodings.append(bgEncoding)
names.append("Brian")

def printClasses():
    """
    Take in current Frame and classify it
    giving the likely probability
    """
    img = Image.fromarray(frame)
    probs = forward_model.predict(img)
    ##find the most likely classes and their probabilities
    topk = list(probs.argsort()[-10:][::-1])
    topprobs = probs[topk]
    imgClass = [Labels[i][:15] for i in topk]
    ##print the MOST likely class and its probability
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
        
        ###I give up just gonna loop through this shit, will work just as well
        for i in range(len(knownEncodings)):
            f = face_recognition.compare_faces(knownEncodings[i], unknownEncoding)
            if(f[0]):
                print ("This is " + names[i])
            else:
                print ("This is not " + names[i])
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

    #classify frame every second by setting up a scheduler
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
