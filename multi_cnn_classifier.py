
#-*- coding: utf-8 -*-
from function_smote import MUL_CNN_preprocess, MUL_CNN
import os
#only dispay warning and error
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'
import tensorflow as tf
from keras import backend as K

n_dim = 500

#get the file path
x_path = 'This is the path of bug reports to be classified'
y_path = 'This is the path of labels of bug reports'

model_path = 'This is the path of the semantic model'
save_path = 'This is the path of result saving file'

x, y, y_kinds=MUL_CNN_preprocess(n_dim,x_path, y_path, model_path, step_rate=0.2)

f_save = open(save_path,'w')
for i in range(100):
	print (i)
	#try:	
	result=MUL_CNN(n_dim,x, y, y_kinds, batch_size=512, n_filter=32,
        filter_length=5, nb_epoch=50, n_pool=3)
	f_save.write(result+'\n')
	K.clear_session()
	tf.reset_default_graph()
	#except:
	#    continue
f_save.close()
