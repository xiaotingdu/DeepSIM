
#-*- coding: utf-8 -*-
from function_new import MUL_CNN_preprocess, MUL_CNN
import os
#only dispay warning and error
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'
import tensorflow as tf
from keras import backend as K
#window = 5 
#min_count=5
n_dim = 500

#get the file path
x_path = '../bugreports_classification/linux1/summary/multiclass_linux1_arb_s.txt'
y_path = '../bugreports_classification/linux1/summary/multiclass_linux1_arb_label_s.txt'

model_path = '../model_dxt/dimension/bugreport-vectors-gensim-sg500d_5w_5m.bin'
save_path = '../output/word2vec_cnn/linux1/linux1_cnn_arbsubtypes_s.txt'

x, y, y_kinds=MUL_CNN_preprocess(n_dim,x_path, y_path, model_path, step_rate=0.2)

f_save = open(save_path,'w')
for i in range(100):
	print (i)
	#try:	
	result=MUL_CNN(n_dim,x, y, y_kinds, batch_size=512, n_filter=32,
        filter_length=5, nb_epoch=50, n_pool=3)
	#MUL_CNN(dimension,x, y, y_kinds, batch_size=64, n_filter=32,
    #    filter_length=5, nb_epoch=15, n_pool=3)
	print (result)
	# CNN(x_dataset, y_dataset, batch_size=64, n_filter=32,
	#filter_length=5, nb_epoch=15, n_pool=3):
	#batch_size：每次送进网络训练的数据数量
	#n_filter：训练所用的卷积核的数目
	#filter_length:卷积核的宽度（长度为350，默认为单词的向量长度）
	#nb_epoch:数据训练的总轮数
	#n_pool=3：池化层的宽度
	f_save.write(result+'\n')
	K.clear_session()
	tf.reset_default_graph()
	#except:
	#    continue
f_save.close()