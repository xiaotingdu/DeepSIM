#-*- coding: utf-8 -*-
from function_new import CNN_preprocess, CNN
import os
import numpy as np
#only dispay warning and error
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'
import tensorflow as tf
from keras import backend as K

#get the file path
pos_path='D:/Experiments/bugreports_classification/linux1/summary/linux1_bug_s.txt'
neg_path='D:/Experiments/bugreports_classification/linux1/summary/linux1_nonbug_s.txt'

#window = 5 
#min_count=5
n_dim = 500

print ('Starting...')
#model_path = '../model_dxt/dimension/bugreport-vectors-gensim-sg500d_5w_5m.bin'
model_path = 'D:/Experiments/model_dxt/random/bugreport-vectors-gensim-sg500d_5w_5m_0.2.bin'
#model_path = '../model_wiki/word_linear_sg_500d/words500.npy'
#model_path = '../model_google/GoogleNews-vectors-negative300.bin'
#save_path = '../output/word2vec_cnn/linux1/linux1_cnn_bug_nonbug_s.txt'
save_path = 'D:/Experiments/output/random/linux1_cnn_bug_nonbug_s_0.2.txt'
print (model_path)
print (save_path)

x_dataset, y_dataset=CNN_preprocess(n_dim, pos_path, neg_path, model_path)

f_save = open(save_path,'w')
for i in range(100):
	print (i)
	#try:	
	result=CNN(n_dim,x_dataset, y_dataset, batch_size=512, n_filter=32,
        filter_length=5, nb_epoch=50, n_pool=3)
	#CNN(dimension,x_dataset, y_dataset, batch_size=64, n_filter=16,
    #    filter_length=5, nb_epoch=15, n_pool=3)  #Bug/Nonbug;BOH/MAN
	#print (result)
	#batch_size：每次送进网络训练的数据数量
	#n_filter：训练所用的卷积核的数目
	#filter_length:卷积核的宽度（长度为350，默认为单词的向量长度）
	#nb_epoch:数据训练的总轮数
	#n_pool=3：池化层的宽度
	f_save.write(result+'\n')
	K.clear_session()
	tf.reset_default_graph()
	#except:
	#	continue
f_save.close()
