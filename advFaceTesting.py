import image_processing
import face_recognition
import PIL
import numpy
import cv2
import time
import image_processing

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
#set up for Sd
sdImage = face_recognition.load_image_file("media/Sd.png")
sdEncoding = face_recognition.face_encodings(sdImage)
knownEncodings.append(sdEncoding)
names.append("Sd")
#set up for Brad Pitt
bpImage = face_recognition.load_image_file("media/BradPitt.png")
bpEncoding = face_recognition.face_encodings(bpImage)
knownEncodings.append(bpEncoding)
names.append("Brad Pitt")
#set up for Milla Jovovich
mjImage = face_recognition.load_image_file("media/MillaJovovich.png")
mjEncoding = face_recognition.face_encodings(mjImage)
knownEncodings.append(mjEncoding)
names.append("Milla Jovovich")

def faceChecker(frame, tolerance):
    """
    Take an image(as numpy array) and check with list of known faces
    print off results for simplicity
    """
    #display image first, colors are weird because cv2 uses BGR lol
    #cv2.imshow('image', frame)
    #cv2.waitKey(0)
    
    #get encoding for image
    if(len(face_recognition.face_encodings(frame)) > 0):
        unknownEncoding = face_recognition.face_encodings(frame)[0]

        match = False
        ###I give up just gonna loop through this shit, will work just as well
        for i in range(len(knownEncodings)):
            f = face_recognition.compare_faces(knownEncodings[i], unknownEncoding, tolerance)
            if(f[0]):
                print ("This is " + names[i])
                print (face_recognition.face_distance(knownEncodings[i], unknownEncoding))
                match = True
        if(not match):
            print("No Matches")
        #testing OW adversarial detecting
        print (face_recognition.face_distance(owEncoding, unknownEncoding))

    else:
        print("no face here")

def advProcessingTest(img):
    """
    Take a PIL image, perform transforms on it, and compare
    face distances of transformed images
    """
    
    col_img = image_processing.color_shift(img)
    sat_img = image_processing.saturate_mod(img)
    noise_img = image_processing.add_noise(img)
    warp_img = image_processing.rand_warp(img)

    imArr = numpy.array(img)
    imEnc = face_recognition.face_encodings(imArr)
    colArr = numpy.array(col_img)
    colEnc = face_recognition.face_encodings(colArr)
    satArr = numpy.array(sat_img)
    satEnc = face_recognition.face_encodings(satArr)
    noiseArr = numpy.array(noise_img)
    noiseEnc = face_recognition.face_encodings(noiseArr)
    warpArr = numpy.array(warp_img)
    warpEnc = face_recognition.face_encodings(warpArr)

    print("\nColor Shift")
    print(face_recognition.face_distance(imEnc[0], colEnc))
    print("\nSaturation")
    print(face_recognition.face_distance(imEnc[0], satEnc))
    print("\nAdd Noise")
    print(face_recognition.face_distance(imEnc[0], noiseEnc))
    print("\nWarp")
    print(face_recognition.face_distance(imEnc[0], warpEnc))

def checkAdversarial(img):
    """
    Takes in PIL image and checks for adversarial, the algorithm is
    loosely as follows: see what face (or not) is recognized before
    image transformations and after image transformations
    Lean towards after transformations being the correct choice
    """
    
    col_img = image_processing.color_shift(img)
    sat_img = image_processing.saturate_mod(img)
    noise_img = image_processing.add_noise(img)
    warp_img = image_processing.rand_warp(img)
    gblur_img = image_processing.gaussian_blur(img, 3)

    print("\nOriginal Image")
    faceChecker(numpy.array(img), .5)    
    print("\nColor Shift")
    faceChecker(numpy.array(col_img), .5)
    print("\nSaturation")
    faceChecker(numpy.array(sat_img), .5)
    print("\nAdd Noise")
    faceChecker(numpy.array(noise_img), .5)
    print("\nWarp")
    faceChecker(numpy.array(warp_img), .5)
    print("\nGaussian Blur")
    faceChecker(numpy.array(gblur_img), .5)
    

    
    
"""
While all of the below didnt quite work, some were rather close upon examining
face_distances
"""
"""
#no success
BGadv = face_recognition.load_image_file("media/BG1adv.png")
faceChecker(BGadv)
#no success
OZadv = face_recognition.load_image_file("media/OZ1adv.png")
faceChecker(OZadv)
#CLOSEST SUCCESS, will use owen wilson as example
OWadv = face_recognition.load_image_file("media/OwenWilson1adv.jpg")
faceChecker(OWadv)
#test sa impersonating sd - FAIL
SAadv = face_recognition.load_image_file("media/Sa.png")
faceChecker(SAadv)
saEncoding = face_recognition.face_encodings(SAadv)
print(face_recognition.face_distance(saEncoding, sdEncoding[0]))
#test sc impersontaing BradPitt - FAIL
SCadv = face_recognition.load_image_file("media/Sc.png")
faceChecker(SCadv)
scEncoding = face_recognition.face_encodings(SCadv)
print(face_recognition.face_distance(scEncoding, bpEncoding[0]))
#test sb impersonating Milla - FAIL
SBadv = face_recognition.load_image_file("media/Sb.png")
faceChecker(SBadv)
sbEncoding = face_recognition.face_encodings(SBadv)
print(face_recognition.face_distance(sbEncoding, mjEncoding[0]))

#CLOSEST SUCCESS, will use owen wilson as example
OWadv = face_recognition.load_image_file("media/OwenWilson1adv.jpg")
faceChecker(OWadv)
owadvEncoding = face_recognition.face_encodings(OWadv)
knownEncodings.append(owadvEncoding)
names.append("OW ADV")

######Owen Wilson adversarial transformation testing

####ON ADVERSARIAL IMAGE - mild perturbations to encodings
print("\n\n  Against Adversarial Image  \n")
img = PIL.Image.open("media/OwenWilson1adv.jpg")
advProcessingTest(img)

####ON ORIGINAL IMAGE - mild perturbations to encodings
print("\n\n  Against Original Image  \n")
img = PIL.Image.open("media/OwenWilson1.jpg")
advProcessingTest(img)

"""

img = PIL.Image.open("media/OwenWilson1adv.jpg")
print("\nAdversarial Image\n")
checkAdversarial(img)

img = PIL.Image.open("media/OwenWilson1.jpg")
print("\nnNormal Image\n")
checkAdversarial(img)



