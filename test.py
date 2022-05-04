import pickle
import cv2
import numpy as np
import glob, os

bias = 0
weights = pickle.load(open( "weights", "rb" ))

img = cv2.imread("input.jpg", 0)
img = cv2.resize(img,(20,20))
cv2.imshow('Input rescaled', img)
cv2.waitKey(0)
activation = ((img / 255.0).flatten())
predictions = ["Cat","Dog"]

list = []
for i in range(400):
    n = activation[i] * weights[i]
    list.append(n)
    out = sum(list)
final = 0
if (out < bias):
    final = 0
elif (out > bias):
    final = 1

print("Out:" + str(out) + " " + predictions[final])
