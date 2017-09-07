import os
import sys
import numpy as np
from copy import deepcopy

neighbours = 3

class FeatureVector(object):
	def __init__(self, vocabsize, numdata):
		self.vocabsize = vocabsize
		self.X = np.empty((0, vocabsize), dtype=int)
		self.Y = np.array([], dtype=int)

	def make_featurevector(self, inputs, classid):
		vector = np.array(inputs.values(), dtype = int)
		vector = np.reshape(vector, (1, self.vocabsize))
		self.X = np.append(self.X, vector, axis = 0)
		self.Y = np.append(self.Y, classid)

class KNN(object):
	def __init__(self,trainVec,testVec):
		self.X_train = trainVec.X
		self.Y_train = trainVec.Y
		self.X_test = testVec.X
		self.Y_test = testVec.Y
		self.metric = Metrics('accuracy')

# Check distances and stuff

	def classify(self, nn=1):
		# trainlen = self.X_train.shape[0]
		# testlen = self.X_test.shape[0]
		preds = []

		for i in xrange(self.X_test.shape[0]):
			qvals = np.ones(neighbours) * -1 * float("inf")
			qinds = np.ones(neighbours, dtype=int) * -1
			for j in xrange(self.X_train.shape[0]):
				distval = np.dot(self.X_test[i], self.X_train[j]) / (float(np.linalg.norm(self.X_test[i]) * np.linalg.norm(self.X_train[j])))

				minind = np.argmin(qvals)
				if qvals[minind] < distval:
					qinds[minind] = j
					qvals[minind] = distval

			votes = np.array([], dtype=int)
			for k in qinds:
				votes = np.append(votes, self.Y_train[k])

			output = np.argmax(np.bincount(votes))
			preds.append(output)

		matrix =  0
		accuracy = 0
		f1score = 0
		precision = 0
		return preds, accuracy, f1score, matrix, precision

def recall(preds, labels, c):
	count = 0
	total = 0
	for i in xrange(len(preds)):
		if preds[i] == c and preds[i] == labels[i]:
			count += 1

	for i in xrange(mylen):
		if labels[i] == c:
			total += 1.0
	return count / float(total)

def precision(preds, labels, c):
	count = 0
	total = 0
	for i in xrange(len(preds)):
		if preds[i] == c and preds[i] == labels[i]:
			count += 1
			total += 1
		elif preds[i] == c and preds[i] != labels[i]:
			total += 1

	if total != 0:
		return count / (float(total + 0.0))
	else:
		return 0


class Metrics(object):
	def __init__(self, metric):
		self.metric = metric

	def get_confmatrix(self, y_pred, y_test):
		length = 10
		matrix = np.zeros((length, length))
		for i in xrange(len(y_pred)):
			indice = y_pred[i] - 1
			indice2 = y_test[i] - 1
			matrix[indice][indice2] += 1

		return matrix

	def accuracy(self, y_pred, y_test):
		count = 0
		for i in xrange(len(y_pred)):
			if y_test[i] == y_pred[i]:
				count += 1
		return count / (float(len(y_pred) + 0.0))

	def f1_score(self, y_pred, y_test):
		score = 0
		for c in xrange(0, 10):
			prec = precision(y_pred, y_test, c + 1)
			rec = recall(y_pred, y_test, c + 1)
			# print(prec, rec)
			num = float(2 * prec * rec * 1.0)
			denom = float((prec + rec) * 1.0)
			if denom == float(0):
				upd = 0
			else:
				upd = 1/(float(10) + 0.0) * (num / denom)
			score += upd
		return score

if __name__ == '__main__':


	train = sys.argv[1]
	test = sys.argv[2]
	classes = ['galsworthy/','galsworthy_2/','mill/','shelley/','thackerey/','thackerey_2/','wordsmith_prose/','cia/','johnfranklinjameson/','diplomaticcorr/']
	cleanclasses = ['galsworthy','galsworthy_2','mill','shelley','thackerey','thackerey_2','wordsmith_prose','cia','johnfranklinjameson','diplomaticcorr']
	inputdir = [train, test]

	vocab = 0
	vocabdict= {}
	trainsize = 0
	testsize = 0

	for idir in inputdir:
		classid = 1
		for c in classes:
			# print(c)
			listing = os.listdir(idir + c)
			for filename in listing:
				if idir == sys.argv[1]:
					trainsize += 1
					if os.stat(str(idir + c + filename)).st_size == 0:
						pass
					else:
						with open(idir + c + filename, 'r') as f:
							for line in f:
								for word in line.split():
									try:
										x = vocabdict[word]
									except:
										if str.isalpha(word):
											vocabdict[word] = 0
											vocab += 1
				else:
					testsize += 1

	vocab += 1
	vocabdict['Unknown'] = 0

	vocablist = list(vocabdict)

	trainVec = FeatureVector(vocab, trainsize)
	testVec = FeatureVector(vocab, testsize)

	for idir in inputdir:
		classid = 1
		for c in classes:
			# print(c)
			listing = os.listdir(idir+c)
			for filename in listing:
				if os.stat(str(idir + c + filename)).st_size == 0 and idir == sys.argv[1]:
					pass
				else:
					myvector = deepcopy(vocabdict)
					f = open(idir+c+filename,'r')
					for line in f:
						for word in line.split():
							try:
								a = vocabdict[word]
								myvector[word] += 1
							except:
								if str.isalpha(word):
									myvector['Unknown'] += 1


				if idir == sys.argv[1]:
					trainVec.make_featurevector(myvector, classid)
				else:
					testVec.make_featurevector(myvector, classid)

			classid += 1

	# print('Finished making features.')

	# print(trainVec.X.shape, trainVec.Y.shape, testVec.X.shape, testVec.Y.shape)

	knn = KNN(trainVec, testVec)
	parameters = knn.classify()
	outpreds = parameters[0]
	accuracy = parameters[1]
	f1score = parameters[2]
	conf_matrix = parameters[3]
	precision = parameters[4]

	for i in outpreds:
		print cleanclasses[i - 1]
	# print accuracy

	# print conf_matrix
