import image_processing
import face_recognition
from PIL import Image
import numpy
import cv2
import time

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
#set up for Owen Wilson
owImage = face_recognition.load_image_file("media/OwenWilson1.jpg")
owEncoding = face_recognition.face_encodings(owImage)
knownEncodings.append(owEncoding)
names.append("Owen Wilson")

def faceChecker(frame):
    """
    Take an image and check with list of known faces
    print off results for simplicity
    """
    #display image first, colors are weird because cv2 uses BGR lol
    cv2.imshow('image', frame)
    cv2.waitKey(0)
    
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

#no success
BGadv = face_recognition.load_image_file("media/BG1adv.png")
faceChecker(BGadv)
#no success
OZadv = face_recognition.load_image_file("media/OZ1adv.png")
faceChecker(OZadv)
#no success
OWadv = face_recognition.load_image_file("media/OwenWilson1adv.jpg")
faceChecker(OWadv)
