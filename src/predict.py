# -*- coding: utf-8 -*-
import os, sys, pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np

class Recommender:
	doc_weights, ids = None, None

	def __init__(self, modelFile, idFile):
		# LOAD MODEL AND LABELS
		model = tf.keras.models.load_model(modelFile)
		self.ids = np.load(idFile, allow_pickle=True)
		doc_layer = model.get_layer('doc_embedding')
		self.doc_weights = doc_layer.get_weights()[0]
		self.doc_weights = self.doc_weights / np.linalg.norm(self.doc_weights, axis = 1).reshape((-1, 1))

	def run(self, doc):
		if not doc in self.ids.values:
			print ("Cannot find document!")
			return []

		docNum = self.ids[self.ids == doc].index[0]

		# PREDICTION
		dists = np.dot(self.doc_weights, self.doc_weights[docNum])
		sorted_ids = np.argsort(dists)
		sorted_dists = sorted(dists)[-11:]
		closest_ids = sorted_ids[-11:]

		result = []
		for i in range(len(closest_ids)):
			result.append((self.ids.values[closest_ids[i]], sorted_dists[i].astype(float)))

		result.reverse()
		return result


if __name__ == '__main__':
	modelFile = sys.argv[1]
	idFile = sys.argv[2]
	document = sys.argv[3]

	r = Recommender(modelFile, idFile)
	for r in r.run(document):
		print (r)