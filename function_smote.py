import tensorflow as tf
from keras import backend as K
from gensim.models.keyedvectors import KeyedVectors
from sklearn.metrics import confusion_matrix
import numpy as np
from keras.layers import Convolution1D, MaxPooling1D
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.models import Sequential
from sklearn import metrics
from sklearn.model_selection import train_test_split


def WordProcess(text):
	reports_list = []
	for report in text:
		report_list = []
		for x in report.replace(' \n', '').split(' '):
			report_list.append(x)
		reports_list.append(report_list)
	return reports_list


def LabelProcess(y):
	y_kind = list(set(y))
	y_kinds = []
	for w in y_kind:
		y_kinds.append(w[:3])
	y_kinds = list(set(y_kinds))
	encode = np.eye(len(y_kinds))
	for i in range(len(y_kinds)):
		print(y_kinds[i], ':encode is', encode[i], '; NUM:', i)
	return y_kinds, encode


def load_reports(path):
	with open(path, 'r', encoding='UTF-8') as infile:
		reports = infile.readlines()
	return reports


def load_model(model_path):
	print("loading model")
	#model = KeyedVectors.load_word2vec_format(model_path)
	model = KeyedVectors.load_word2vec_format(model_path, binary=True)
	print("load complete")
	return model


def max_num(report):
	max = 0;
	for i in range(len(report)):
		temp = len(report[i])
		if (temp > max):
			max = temp
	return max


def combine(report1, report2):
	corpus = np.concatenate((report1, report2))
	return corpus


def build_matrix(dimension,x, padding_size, model):
	res = []
	for sen in x:
		matrix = []
		for i in range(padding_size):
			try:
				#print (model[sen[i]])
				matrix.append(model[sen[i]])
			except:
				# 这里有两种except情况，
				# 1. 这个单词找不到
				# 2. sen没那么长
				# 不管哪种情况，我们直接贴上全是0的vec
				matrix.append([0] *dimension)
		#print (type(matrix))
		res.append(matrix)
	#print (type(res))
	return res

def y_dataset_process(y):
	"""
    process y into [0,1]
    """
	y_processed = np.zeros([y.shape[0], 2])
	for i in range(len(y)):
		if (y[i] == 1):
			y_processed[i] = [1, 0]  # boh
		else:
			y_processed[i] = [0, 1]  # man
	return y_processed


def decide(y_pre):
	row, column = y_pre.shape
	for i in range(row):
		if y_pre[i, 0] > y_pre[i, 1]:
			y_pre[i, 0] = 1
			y_pre[i, 1] = 0
		else:
			y_pre[i, 0] = 0
			y_pre[i, 1] = 1
	return y_pre


def score(y_pre, y_test):
	cm = confusion_matrix(y_test, y_pre)
	tn = cm[0][0]
	fp = cm[0][1]
	fn = cm[1][0]
	tp = cm[1][1]
	# accuracy
	accuracy = 1.0 * (tp + tn) / (tp + tn + fp + fn)
	
	# precision_p
	precision_p = 1.0 * tp / (tp + fp)
	
	# precision_n
	precision_n = 1.0 * tn / (tn + fn)
	# recall_p
	recall_p = 1.0 * tp / (tp + fn)
	
	# recall_n
	recall_n = 1.0 * tn / (tn + fp)
	
	# f1_score_p
	f1_score_p = 2.0 * (precision_p * recall_p) / (precision_p + recall_p)
	
	# f1_score_n
	f1_score_n = 2.0 * (precision_n * recall_n) / (precision_n + recall_n)
	
	# average_precision
	average_precision = (1.0 * precision_n * sum(cm[:, 0]) + 1.0 * precision_p * sum(cm[:, 1])) / (
				sum(cm[0, :]) + sum(cm[1, :]))
	
	average_recall = (1.0 * recall_n * sum(cm[:, 0]) + 1.0 * recall_p * sum(cm[:, 1])) / (sum(cm[0, :]) + sum(cm[1, :]))
	
	average_f1_score = (1.0 * f1_score_n * sum(cm[:, 0]) + 1.0 * f1_score_p * sum(cm[:, 1])) / (
				sum(cm[0, :]) + sum(cm[1, :]))
	
	# print("accuracy=%.4f" % accuracy)
	# print("precision_p=%.4f" % precision_p)
	# print("precision_n=%.4f" % precision_n)
	# print("recall_p=%.4f" % recall_p)
	# print("recall_n=%.4f" % recall_n)
	# print("f1_score_p=%.4f" % f1_score_p)
	# print("f1_score_n=%.4f" % f1_score_n)
	# print("average_precision=%.4f" % average_precision)
	# print("average_recall=%.4f" % average_recall)
	# print("average_f1_score=%.4f" % average_f1_score)
	
	result = str("%.4f" % accuracy) + '\t' + str("%.4f" % precision_p) + '\t' + str("%.4f" % recall_p) + '\t' + str(
		"%.4f" % f1_score_p) + '\t' + str("%.4f" % precision_n) + '\t' + str("%.4f" % recall_n) + '\t' + str(
		"%.4f" % f1_score_n) + '\t' + str("%.4f" % average_precision) + '\t' + str(
		"%.4f" % average_recall) + '\t' + str("%.4f" % average_f1_score)
	return result
def deprocess(y):
	"""
    deprocess y into [1]
    """
	y_deprocessed = np.zeros([y.shape[0], 1])
	for i in range(len(y)):
		if (y[i] == [1, 0]).all():
			y_deprocessed[i] = 1
		else:
			y_deprocessed[i] = 0
	return y_deprocessed


#def data_expend(x_pos, x_neg, step_rate):
	# if (len(x_pos) > len(x_neg)):
	# 	short = x_neg
	# 	rate = round(len(x_pos) / len(x_neg)) - 1
	# 	for i in range(rate):
	# 		r = np.random.random((short.shape[0], short.shape[1], short.shape[2]))
	# 		r = (r * 2 - 1) / (1/step_rate)
	# 		x_neg = np.concatenate((x_neg, short + np.multiply(short, r)))
	# if (len(x_neg) > len(x_pos)):
	# 	short = x_pos
	# 	rate = round(len(x_neg) / len(x_pos)) - 1
	# 	for i in range(rate):
	# 		r = np.random.random((short.shape[0], short.shape[1], short.shape[2]))
	# 		r = (r * 2 - 1) /(1/step_rate)
	# 		x_pos = np.concatenate((x_pos, short + np.multiply(short, r)))
		# x_pos=np.concatenate((x_pos, short+r))
	#    if(len(x_pos)<200 or len(x_neg)<200):
	#        temp_pos=x_pos
	#        temp_neg=x_neg
	#        rate=round(200/len(x_pos))
	#        for i in range(rate):
	#            r_pos=np.random.random((temp_pos.shape[0], temp_pos.shape[1], temp_pos.shape[2]))
	#            r_neg=np.random.random((temp_neg.shape[0], temp_neg.shape[1], temp_neg.shape[2]))
	#            r_pos=(r_pos*2-1)/10
	#            r_neg=(r_neg*2-1)/10
	#            x_pos=np.concatenate((x_pos, temp_pos+np.multiply(temp_pos, r_pos)))
	#            x_neg=np.concatenate((x_neg, temp_neg+np.multiply(temp_neg, r_neg)))
	##            x_pos=np.concatenate((x_pos, temp_pos+r_pos))
	##            x_neg=np.concatenate((x_neg, temp_neg+r_neg))
	# return x_pos, x_neg

def CNN_preprocess(dimension,pos_path, neg_path, model_path, step_rate=0.2):
	pos_reports = load_reports(pos_path)
	neg_reports = load_reports(neg_path)
	model_word = load_model(model_path)
	# process the reports
	pos_reports = WordProcess(pos_reports)
	neg_reports = WordProcess(neg_reports)
	corpus = combine(pos_reports, neg_reports)
	# find the max_len in corpus
	padding_size = min(max_num(corpus),30)
    
	# get the matrix of reports
	x_pos = build_matrix(dimension,pos_reports, padding_size, model_word)
	x_neg = build_matrix(dimension,neg_reports, padding_size, model_word)
	x_pos = np.array(x_pos)
	x_neg = np.array(x_neg)
	# balance the num of neg and pos
	# x_pos, x_neg = data_expend(x_pos, x_neg,step_rate)
	# generate labels
	y_pos = np.ones((len(x_pos), 1))
	y_neg = np.zeros((len(x_neg), 1))
	# generate dataset
	x_dataset = np.concatenate((x_pos, x_neg))
	y_dataset = np.concatenate((y_pos, y_neg))
	y_dataset = np.array(y_dataset)
	y_dataset = y_dataset.reshape((len(y_dataset), 1))
	x_dataset = np.array(x_dataset)

	return x_dataset, y_dataset, padding_size
def MUL_CNN_preprocess(dimension,x_path, y_path, model_path, step_rate=0.2):

	x_reports = load_reports(x_path)
	y_reports = load_reports(y_path)
	model_word= load_model(model_path)
	#处理掉多余的字节
	y_reports = [w[:3] for w in y_reports]
	#处理x_reports
	x_reports = WordProcess(x_reports)
	#find the max_len in x_reports
	padding_size = min(max_num(x_reports),50)
	#build bug matrix(padding_size*350)
	x = build_matrix(dimension,x_reports, padding_size, model_word)
	# get the y_kinds
	y_kinds, encode = LabelProcess(y_reports)#encode 是不同种类的编码
	# balance the data
	# num = np.zeros([len(y_kinds), 1])#存放不同种类的数目
	# rate = np.zeros([len(y_kinds), 1])#存放需要扩展的比率
	#
	# for i in range(len(y_kinds)):
	# 	num[i] = y_reports.count(y_kinds[i])
	#
	# maxnum = num.max()
	# # rate
	# for i in range(len(y_kinds)):
	# 	rate[i] = round(maxnum / num[i][0]) - 1
	
	# y
	y = []
	for w in y_reports:
		y.append(y_kinds.index(w))
	#
	x=np.array(x)
	#### over_sampling
	from imblearn.over_sampling import SMOTE
	smo = SMOTE()
	
	x_dup = []
	for matrix in x:
		x_dup.append(np.reshape(matrix, (padding_size * dimension)))
	x_dup = np.array(x_dup)
	x_exp, y_exp = smo.fit_sample(x_dup, y)
	
	x = []
	# x_dataset
	for matrix in x_exp:
		x.append(np.reshape(matrix, (padding_size, dimension)))
	x = np.array(x)
	
	#encode
	y=[]
	for w in y_exp:
		y.append(encode[w])
	y = np.array(y)
	# Expend
	# for i in range(len(y)):
	# 	index = np.argmax(y[i])
	# 	expend_rate = rate[index][0]
	# 	for j in range(int(expend_rate)):
	# 		y.append(y[i])
	# 		r = np.random.random((len(x[1]), dimension))
	# 		r = (r * 2 - 1)/(1/step_rate)
	# 		temp = x[i] + np.multiply(x[i], r)
	# 		x.append(temp)
	#
	# y = np.array(y)
	# x = np.array(x)
	return x, y, y_kinds
def CNN(dimension,x_dataset, y_dataset, batch_size=64, n_filter=32,
        filter_length=3, nb_epoch=15, n_pool=3,padding_size = 10 ):
	
	# generate the train and test dataset
	x_train, x_test, y_train, y_test = train_test_split(x_dataset, y_dataset, test_size=0.2)
	#### over_sampling
	result_list = []
	from imblearn.over_sampling import SMOTE
	smo = SMOTE()
	
	x_train_dup = []
	for matrix in x_train:
		x_train_dup.append(np.reshape(matrix, (x_train.shape[1] * x_train.shape[2])))
	
	x_train_exp, y_train_exp = smo.fit_sample(x_train_dup, y_train[:, 0])
	
	x_train = []
	y_train = []
	# y_train
	for i in range(len(y_train_exp)):
		if (y_train_exp[i] == 1):
			y_train.append([1, 0])
		elif (y_train_exp[i] == 0):
			y_train.append([0, 1])
	y_train = np.array(y_train)
	# x_dataset
	for matrix in x_train_exp:
		x_train.append(np.reshape(matrix, (padding_size, dimension)))
	x_train = np.array(x_train)

	# build CNN
	#print (x_train.shape[0])
	#print (x_train.shape[1])
	#print (len(x_train))
	model = Sequential()
	print (x_train.shape[1])
	model.add(Convolution1D(n_filter, filter_length,
	                        input_shape=(x_train.shape[1], dimension)))
	model.add(Activation('tanh'))
	model.add(Convolution1D(n_filter, filter_length))
	model.add(Activation('tanh'))
	model.add(MaxPooling1D(pool_size=(n_pool)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	
	# Ann
	model.add(Dense(dimension))
	model.add(Activation('sigmoid'))
	model.add(Dropout(0.5))
	model.add(Dense(2))
	model.add(Activation('softmax'))
	
	# compile
	model.compile(loss='mse',
	              optimizer='Adam',
	              metrics=['accuracy'])
	
	# get result
	model.fit(x_train, y_train, batch_size=batch_size,
	          epochs=nb_epoch, verbose=0)
	y_pre = model.predict(x_test)
	y_pre = decide(y_pre)
	y_pre = deprocess(y_pre)
	y_test =deprocess(y_test)
	result=score(y_pre, y_test)
	K.clear_session()
	tf.reset_default_graph()
	return result

def MUL_CNN(dimension,x, y, y_kinds, batch_size=64, n_filter=32,
        filter_length=5, nb_epoch=15, n_pool=3):
	
	#get the train and test dataset
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
	# sequential model
	model = Sequential()
	model.add(Convolution1D(n_filter, filter_length,
	                        input_shape=(x_train.shape[1], dimension)))
	model.add(Activation('tanh'))
	model.add(Convolution1D(n_filter, filter_length))
	model.add(Activation('tanh'))
	model.add(MaxPooling1D(pool_size=(n_pool)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	
	# Ann
	model.add(Dense(160))
	model.add(Activation('sigmoid'))
	model.add(Dropout(0.5))
	model.add(Dense(len(y_kinds)))
	model.add(Activation('softmax'))
	
	# compile
	model.compile(loss='mse',
	              optimizer='Adam',
	              metrics=['accuracy'])
	model.fit(x_train, y_train, batch_size=batch_size,
	          epochs=nb_epoch, verbose=0)
	y_pre = model.predict(x_test)
	#分配空间
	y_pre_processed = np.ones([len(y_pre), 1])
	y_test_processed = np.ones([len(y_pre), 1])
	#转换到一维
	for i in range(len(y_pre)):
		y_pre_processed[i] = np.argmax(y_pre[i])
		y_test_processed[i] = np.argmax(y_test[i])
	#计算结果
	precision = metrics.precision_score(y_test_processed, y_pre_processed, average=None)
	recall = metrics.recall_score(y_test_processed, y_pre_processed, average=None)
	cm = confusion_matrix(y_test_processed, y_pre_processed)
	accuracy = (np.mean(y_pre_processed == y_test_processed))
	f1_score = metrics.f1_score(y_test_processed, y_pre_processed, average=None)
	#计算平均指标
	precision_avg = np.sum(np.multiply(np.sum(cm, 1), precision)) / np.sum(cm)
	
	recall_avg = np.sum(np.multiply(np.sum(cm, 1), recall)) / np.sum(cm)
	
	f1_avg = np.sum(np.multiply(np.sum(cm, 1), f1_score)) / np.sum(cm)
	result = str("%.4f" % accuracy)
	for i in range(len(y_kinds)):
		result = result + '\t' + str("%.4f" % precision[i]) + '\t' + str("%.4f" % recall[i]) + '\t' + str("%.4f" % f1_score[i])
	result = result + '\t' + str("%.4f" % precision_avg) + '\t' + str("%.4f" % recall_avg) + '\t' + str("%.4f" % f1_avg)
	K.clear_session()
	tf.reset_default_graph()
	return result
