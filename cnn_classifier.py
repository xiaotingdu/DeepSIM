#-*- coding: utf-8 -*-
from function_smote import CNN_preprocess, CNN
import os
import numpy as np
#only dispay warning and error
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'
import tensorflow as tf
from keras import backend as K

#get the file path
pos_path='Please insert the path of the pos class here'
neg_path='Please insert the path of the neg class here'

n_dim = 500
model_path = 'This is the path of the semantic model'
save_path = 'This is the results saving path'


x_dataset, y_dataset=CNN_preprocess(n_dim, pos_path, neg_path, model_path)

f_save = open(save_path,'w')
for i in range(100):
	print (i)
	#try:	
	result=CNN(n_dim,x_dataset, y_dataset, batch_size=512, n_filter=32,
        filter_length=5, nb_epoch=50, n_pool=3)
	f_save.write(result+'\n')
	K.clear_session()
	tf.reset_default_graph()
	#except:
	#	continue
f_save.close()
