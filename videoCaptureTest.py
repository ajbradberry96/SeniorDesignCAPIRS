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

Labels = forward_model.get_imagenet_labels()
##Using the frame as a global variable because problems passing
##parameters with scheduler
frame = np.empty(1)

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

    #operations on frame, classify and plot
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #classify frame every second
    if(first):
        sched.add_job(printClasses, 'interval', seconds = 5)
        first = False
    
    #display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#when done release everything
cap.release()
cv2.destroyAllWindows()
sched.shutdown()
