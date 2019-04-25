# SeniorDesignCAPIRS
Counter Adversarial Package for Image Recognition Systems - Senior Design LA Tech Winter - Spring 2019

This software seeks to protect image recognition systems from adversarial attacks without having to redesign your entire sysytem. Adversarial attacks can fool just about any image recognition system. Fortunately, this package works with just about any image recognition system as well.

## Installing
This package can be installed with PIP. Try the following command:

`pip3 install capirs`

## Usage
You are going to need a function in your code that can take in an image (in PIL format) and return a vector corresponding to the probabilities of classes that image could be. For instance, if this was an image classifier to classify digits (like MNIST), then that vector might look something like `[0.01, 0.02, 0.01, 0.01, 0.01, 0.03, 0.01, 0.01, 0.97, 0.02]` for the digit "8" (assuming that the elements correspond to the probabilities of the digits 0-9). 

Obviously, this function might look different for any given implementation, but generally any framework should have a way to get logits and probabilities from the classifier. 

Okay, let's assume you made this function and called it predict_image. To detect if an image is an adversarial example, just use the following code.

```
from capirs import detect_adv
detect_adv.predict = predict_img
is_adversarial = detect_adv.detect(img)
```

is_adversarial should be True for adversarial images and False for normal images.

If you are having problems with our implementation being too sensitive or too specific, you can adjust the threshold we use to call differences with the following

`detect_adv.threshold = 0.05`


# Prerequisites
These are the prereqs you need to run all the code listed here. All the dependencies should be handled for you if you just want the package from pip though.

Python 3.6
TenserFlow, open-source machine learning to model the neural network
Pillow, fork of the Python Imaging Library
Numpy for vector math and image modifications
Scipy for spatial vector mathematics and adversarial comparisons
Matplotlib for MATLAB plots
Facial_recognition for face recognition testing (https://github.com/ageitgey/face_recognition)

# Authors
Andrew Bradberry
Brian Greber
Drew Harbor
Joshua Osborne
Christopher Rodriguez
