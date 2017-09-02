import numpy as np
import pandas as pd
import random
import sys

def predict(features, weights, bias):
	activation = 0
	activation = np.matmul(features, np.transpose(weights))
	if activation < 0:
		return 0
	else:
		return 1

def predict_margin(features, weights, bias):
	activation = 0
	activation = np.matmul(features, np.transpose(weights))
	if activation < 0:
		return 0
	else:
		return 1

random.seed(10)

np_df = np.genfromtxt(sys.argv[1], delimiter = ',')
test_set = np.genfromtxt(sys.argv[2], delimiter = ',')


# np_df = df.as_matrix()
# test_set = test_set.as_matrix()

# print np_df.shape
labels = np_df[:, 0]
# print 'labels: ', labels.shape
np_df = np.dot(np_df[:, 1:], 1/255.0)
test_set = np.dot(test_set, 1/255.0)

bias = random.random()
weights = np.random.rand(784)

for k in xrange(20):
	for i in xrange(len(np_df)):
		res = predict(np_df[i], weights, bias)
		if not res == labels[i]:
			if labels[i] == 0:
				temp = np.multiply(0.1, np_df[i])
				weights = np.subtract(weights, temp)
			else:
				temp = np.multiply(0.1, np_df[i])
				weights = np.add(weights, temp)
	# if k % 10 == 0:
	# 	count = 0
	# 	for i in xrange(len(np_df)):
	# 		res = predict(np_df[i], weights, bias)
	# 		if res == labels[i]:
	# 			count += 1

		# print 'Accuracy:', float(count) / len(np_df)

# count = 0
# for i in xrange(len(np_df)):
# 	res = predict(np_df[i], weights, bias)
# 	if res == labels[i]:
# 		count += 1

# print 'Accuracy train:', float(count) / len(np_df)

# test = pd.read_csv('./datasets/MNIST_data_updated/mnist_test.csv')
# test = test.as_matrix()
# new_labels = test[:, 0]
# test = np.dot(test[:, 1:], 1/255.0)

# count = 0
# for i in xrange(len(test)):
# 	res = predict(test[i], weights, bias)
# 	if res == new_labels[i]:
# 		count += 1

# print 'Accuracy normal:', float(count) / len(test)

# print len(test_set[0])

# print test_set.shape

for i in xrange(len(test_set)):
	print(predict(test_set[i], weights, bias))

weights = np.random.rand(784)

for k in xrange(20):
	for i in xrange(len(np_df)):
		res = predict_margin(np_df[i], weights, bias)
		if not res == labels[i]:
			if labels[i] == 0:
				if np.matmul(np_df[i], np.transpose(weights)) > 1:
					temp = np.multiply(0.1, np_df[i])
					weights = np.subtract(weights, temp)
			else:
				if np.matmul(np_df[i], np.transpose(weights)) < -1:
					temp = np.multiply(0.1, np_df[i])
					weights = np.add(weights, temp)

# count = 0
# for i in xrange(len(test)):
# 	res = predict_margin(test[i], weights, bias)
# 	if res == new_labels[i]:
# 		count += 1

# print 'Accuracy margin:', float(count) / len(test)

for i in xrange(len(test_set)):
	print(predict(test_set[i], weights, bias))

weights = np.random.rand(784)
temp = np.multiply(0, np_df[0])

for k in xrange(20):
	for i in xrange(len(np_df)):
		res = predict_margin(np_df[i], weights, bias)
		if not res == labels[i]:
			if labels[i] == 0:
				temp = np.add(temp, np.multiply(-0.1, np_df[i]))
			else:
				temp = np.add(temp, np.multiply(0.1, np_df[i]))
		if i % 30 == 0:
			weights = np.add(weights, temp)
			temp = np.multiply(0, temp)

# count = 0
# for i in xrange(len(test)):
# 	res = predict_margin(test[i], weights, bias)
# 	if res == new_labels[i]:
# 		count += 1

# print 'Accuracy batch without margin:', float(count) / len(test)

for i in xrange(len(test_set)):
	print(predict(test_set[i], weights, bias))

weights = np.random.rand(784)
temp = np.multiply(0, np_df[0])

for k in xrange(20):
	for i in xrange(len(np_df)):
		res = predict_margin(np_df[i], weights, bias)
		if not res == labels[i]:
			if labels[i] == 0:
				if np.matmul(np_df[i], np.transpose(weights)) > 1:
					temp = np.add(temp, np.multiply(-0.1, np_df[i]))
			else:
				if np.matmul(np_df[i], np.transpose(weights)) < -1:
					temp = np.add(temp, np.multiply(0.1, np_df[i]))
		if i % 30 == 0:
			weights = np.add(weights, temp)
			temp = np.multiply(0, temp)

# count = 0
# for i in xrange(len(test)):
# 	res = predict_margin(test[i], weights, bias)
# 	if res == new_labels[i]:
# 		count += 1

# print 'Accuracy batch with margin:', float(count) / len(test)

for i in xrange(len(test_set)):
	print(predict(test_set[i], weights, bias))