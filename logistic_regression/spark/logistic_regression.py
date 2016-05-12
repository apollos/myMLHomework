'''

Created on May 11, 2016

@author: Song Yu

'''
from __future__ import print_function

import sys

from pyspark import SparkContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.linalg import _convert_to_vector
from numpy import *


def parsePoint(line):
    """
    Parse a line of text into an MLlib LabeledPoint object.
    """
    values = [float(s) for s in line.split('\t')]
    size = len(values)   
    return LabeledPoint(values[size-1], values[0:size-1])

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: logistic_regression <trainfile><testfile> <iterations>", file=sys.stderr)
        exit(-1)
    sc = SparkContext(appName="PythonLR-Sy")
    points = sc.textFile(sys.argv[1]).map(parsePoint)
    testPoints = sc.textFile(sys.argv[2]).map(parsePoint)
    iterations = int(sys.argv[3])
    model = LogisticRegressionWithSGD.train(points, iterations)
    print("Final weights: " + str(model.weights))
    print("Final intercept: " + str(model.intercept))
    #######Verify test data
    predV = model.predict(testPoints.map(lambda p: p.features))
    cks = predV.zip(testPoints.map(lambda p: p.label)).map(lambda (a,b): a - b).filter(lambda a: a!=0).count()    
    print("Predict Result: %f" % (float(cks)/float(predV.count())))
    sc.stop()
