# -*- coding: utf-8 -*-
import os, sys, pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np


modelFile = sys.argv[1]
idFile = sys.argv[2]
document = [sys.argv[3]]

print ("Predicting: '" + document[0] + "'")

# LOAD MODEL AND LABELS
model = tf.keras.models.load_model(modelFile)
ids = np.load(idFile, allow_pickle=True)

if not document[0] in ids.values:
	print ("Cannot find document!")
	sys.exit(1)

docNum = ids[ids == document[0]].index[0]

doc_layer = model.get_layer('doc_embedding')
doc_weights = doc_layer.get_weights()[0]
#doc_weights.shape

doc_weights = doc_weights / np.linalg.norm(doc_weights, axis = 1).reshape((-1, 1))

# PREDICTION
dists = np.dot(doc_weights, doc_weights[docNum])
sorted_dists = np.argsort(dists)
closest = reversed(sorted_dists[-10:])

for d in closest:
    #print (d)
    print (ids.values[d])