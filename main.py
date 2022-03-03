import random
import cv2
import numpy as np
import glob, os
import pickle
from PIL import Image as im

os.chdir("dataset/")

E = 2.718
activation = [1 for i in range(400)]
bias = 0
weights = [0 for i in range(400)]

dataset = []
answers = []

predictions = ["Circle","Rectangle"]

def sigmoid(x):
    return -1/(1+pow(E,x))+0.5


# Load dataset
for file in glob.glob("*.png"):
    img = cv2.imread(file, 0)
    dataset.append((img / 255.0).flatten())
    
    f = open(file[:-3]+"txt","r")
    if f.read() == "1":
        answers.append(1)
    else:
        answers.append(0)
    f.close()

for t in range(100):
    for i in range(len(dataset)):
        activation = dataset[i]
        l = []
        for r in range(400):
            n = activation[r] * weights[r]
            l.append(n)

            out = sum(l)
        
        final = 0
        if (out < bias):   
            final = 0
        elif (out > bias):
            final = 1
        
        if final == answers[i]:
            pass
        elif final == 1 and answers[i] == 0:
            for i in range(400):
                weights[i] = weights[i] - activation[i]
        elif final == 0 and answers[i] == 1:
            for i in range(400):
                weights[i] = weights[i] + activation[i]

        print(predictions[final])


os.chdir("../")
pickle.dump(weights, open("weights.txt", "wb"))
