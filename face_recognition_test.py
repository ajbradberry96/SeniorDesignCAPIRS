import image_processing
import face_recognition
import PIL
import numpy

known_image = face_recognition.load_image_file("media/Oz1.png")
oz_encoding = face_recognition.face_encodings(known_image)[0]

def checkIfMyFace(image_array):
    unknown_encoding = face_recognition.face_encodings(image_array)[0]
    return face_recognition.compare_faces([oz_encoding], unknown_encoding)

def checkFileIsFace(filename):
    unknown = face_recognition.load_image_file(filename)
    return checkIfMyFace(unknown)

#print(checkFileIsFace("media/Oz2.jpg"))
#print(checkFileIsFace("media/Oz3.jpg"))
#print(checkFileIsFace("media/not_oz.jpg"))

def modify(filename):
    workingImage = PIL.Image.open(filename).convert("RGB")

    # convert RGBA -> RGB
    #workingImage.load()
    #background = PIL.Image.new("RGBA", workingImage.size, (255, 255, 255))
    #newWorkingImage = PIL.Image.alpha_composite(background, workingImage)

    workingImage = image_processing.add_noise(workingImage)
    #workingImage.save("media/tmp.png")
    return numpy.array(workingImage)

noisedFace = modify("media/Oz1.png")
print(checkIfMyFace(noisedFace))
noisedFace = modify("media/Oz2.jpg")
print(checkIfMyFace(noisedFace))
