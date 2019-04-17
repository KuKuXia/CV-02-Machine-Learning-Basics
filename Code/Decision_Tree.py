#!/usr/bin/python

""" lecture and example code for decision tree unit """

import sys
from class_vis import prettyPicture, output_image
from prep_terrain_data import makeTerrainData

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
from classifyDT import classify
from sklearn.tree import DecisionTreeClassifier
features_train, labels_train, features_test, labels_test = makeTerrainData()



### the classify() function in classifyDT is where the magic
### happens--fill in this function in the file 'classifyDT.py'!
from sklearn.metrics import accuracy_score
clf1 = DecisionTreeClassifier(min_samples_split=2)
clf2 = DecisionTreeClassifier(min_samples_split=50)
clf1.fit(features_train, labels_train)
clf2.fit(features_train, labels_train)
pred1 = clf1.predict(features_test)
accuracy1 = accuracy_score(pred1, labels_test)
pred2 = clf2.predict(features_test)
accuracy2 = accuracy_score(pred2, labels_test)
print(accuracy1)
print(accuracy2)




#### grader code, do not modify below this line

prettyPicture(clf1, features_test, labels_test)
# output_image("test.png", "png", open("test.png", "rb").read())