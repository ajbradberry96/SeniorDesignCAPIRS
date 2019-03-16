import image_processing
import face_recognition
import PIL
import numpy

# Create encoding for Josh's face
known_image = face_recognition.load_image_file("media/Oz1.png")
oz_encoding = face_recognition.face_encodings(known_image)[0]

def checkIfMyFace(image_array):
    """
    Take a list of images and check to see if the first face in the image is Josh.

    :param image_array: numpy array of images to check.
    :return: boolean True if Josh is the first face, False otherwise.
    """
    # convert list of images to list of encodings, take the first one
    unknown_encoding = face_recognition.face_encodings(image_array)[0]
    # check if that encoding corresponds to my face
    return face_recognition.compare_faces([oz_encoding], unknown_encoding)

def checkFileIsFace(filename):
    """
    Check if file is an image of Josh's face.
    Proxy function for checkIfMyFace.

    :param filename:  name of image file
    :return: boolean True if Josh is the first face in the image, False otherwise
    """
    unknown = face_recognition.load_image_file(filename)
    return checkIfMyFace(unknown)

#print(checkFileIsFace("media/Oz2.jpg"))
#print(checkFileIsFace("media/Oz3.jpg"))
#print(checkFileIsFace("media/not_oz.jpg"))

def modify(filename):
    """
    Take in a file and add some noise to it.
    Used to test whether or not the image recognition works with a little noise.
    Turns out to be successful.

    :param filename: name of image file
    :return: numpy array of modified image
    """
    # load image and convert to RGB to work with image processing
    workingImage = PIL.Image.open(filename).convert("RGB")

    # convert RGBA -> RGB
    #workingImage.load()
    #background = PIL.Image.new("RGBA", workingImage.size, (255, 255, 255))
    #newWorkingImage = PIL.Image.alpha_composite(background, workingImage)

    # add some noise
    workingImage = image_processing.add_noise(workingImage)
    # save it (optional for debugging)
    #workingImage.save("media/tmp.png")
    # convert to numpy array and return
    return numpy.array(workingImage)

# create noisy version of Josh's face
noisedFace = modify("media/Oz1.png")
# see if it identifies correctly
print(checkIfMyFace(noisedFace))
# now test a different version of my face
noisedFace = modify("media/Oz2.jpg")
# see if it identifies correctly
print(checkIfMyFace(noisedFace))
