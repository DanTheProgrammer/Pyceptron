import pickle
import cv2
import numpy as np
import glob, os

bias = 0
weights = pickle.load(open( "weights.txt", "rb" ))

img = cv2.imread("input.png", 0)
activation = ((img / 255.0).flatten())
predictions = ["Circle","Rectangle"]

l = []
for i in range(400):
    n = activation[i] * weights[i]
    l.append(n)
    out = sum(l)
final = 0
if (out < bias):   
    final = 0
elif (out > bias):
    final = 1

print(predictions[final])