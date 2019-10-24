import tensorflow as tf
import os

import jieba.posseg as pseg
from tqdm import tqdm
from six.moves import xrange
import numpy as np

from ClassificationModel import ClassificationModel
from DataGenerater import DataGenerater
from Utils import Utils
from Config import Config
from TermsSearch import TermsSearch

class DepartmentClassification:
	def __init__(self):
		# 读取科室的 索引-词
		self.dep1_index_dict = Utils.read_from_ujson(Config.path_dep1_index_dict)

		# 读取术语的 词-索引
		self.term_word_dict = Utils.read_from_ujson(Config.path_term_word_dict)
		self.total_term = len(self.term_word_dict) + 1

		# 读取分词的 词-索引
		self.jieba_word_dict = Utils.read_from_ujson(Config.path_jieba_word_dict)
		self.total_jieba = len(self.jieba_word_dict) + 1

		self.DELTA = 0.5

		self.epoch = 0

	def init_model(self, model_load=True):
		self.sess = tf.Session()
		self.model = ClassificationModel(self.total_jieba, self.total_term)
		self.sess.run(tf.global_variables_initializer())

		if model_load==True:
			self.check_restore_parameters()

	def __train_train(self, trainSetGenerater):
		for train_idx in tqdm(xrange(trainSetGenerater.batch_num)):				
			x_jieba, x_term, y_dep1, x_jieba_len, x_term_len, max_len_jieba, max_len_term = trainSetGenerater.next_batch()

			fetches = [self.model.loss, self.model.optimizer, self.model.pre_y]

			param_feed = {
				self.model.input_q:x_jieba, self.model.input_a:x_term, self.model.input_y:y_dep1, 
				self.model.seq_len_q:x_jieba_len, self.model.seq_len_a:x_term_len,
				self.model.batch_seq_len_q:max_len_jieba, self.model.batch_seq_len_a:max_len_term,
				self.model.dropout_keep_prob:Config.keep_prob
			}

			#train model
			loss_train, _, prey = self.sess.run(fetches, feed_dict=param_feed)
			
			#print(prey)

		print("loss_train:",loss_train)
		global_step = self.epoch * (trainSetGenerater.batch_num) + train_idx
		path = self.model.saver.save(self.sess, Config.model_path, global_step = global_step)
		print("Saved model checkpoint to {}".format(path))

		return loss_train

	def __train_test(self, testSetGenerater):
		predict_label_list = []
		predict5_label_list = []

		for test_idx in tqdm(xrange(testSetGenerater.batch_num)):
			x_jieba, x_term, y_dep1, x_jieba_len, x_term_len, max_len_jieba, max_len_term = testSetGenerater.next_batch()
			
			fetches = [
				self.model.predict_top, 
				self.model.predict_top1,
				self.model.loss, 
				self.model.label
			]

			param_feed = {
				self.model.input_q:x_jieba, self.model.input_a:x_term, self.model.input_y:y_dep1, 
				self.model.seq_len_q:x_jieba_len, self.model.seq_len_a:x_term_len,
				self.model.batch_seq_len_q:max_len_jieba, self.model.batch_seq_len_a:max_len_term,
				self.model.dropout_keep_prob:1.0
			}

			predict_5, predict_1, loss_test_batch, label = self.sess.run(fetches, feed_dict=param_feed)

			for predict,label_ in zip(predict_1[1],label[1]):
				predict_label_list.append((predict,label_))
			for predict,label_ in zip(predict_5[1],label[1]):
				predict5_label_list.append((predict,label_))

		return predict_label_list, predict5_label_list, loss_test_batch

	def train(self):
		print("*****DepartmentClassification.train*****: training begin...")
		# 1、获取训练和测试数据
		trainSet = Utils.readTrainData(Config.path_trainSet)
		testSet = Utils.readTrainData(Config.path_testSet)

		# 2、创建训练、测试数据生成器
		trainSetGenerater = DataGenerater(trainSet, Config.batch_size)
		testSetGenerater = DataGenerater(testSet, Config.batch_size)

		# 3、训练和测试模型
		for epoch in range(Config.NUM_EPOCHS):
			self.epoch = epoch
			# 3.1、训练模型
			loss_train = 0
			loss_tr = self.__train_train(trainSetGenerater)
			loss_train = loss_tr * self.DELTA + loss_train * (1 - self.DELTA)

			# 3.2、测试模型
			loss_test = 0
			predict_label_list, predict5_label_list, loss_test_batch = self.__train_test(testSetGenerater)
			loss_test += loss_test_batch

			score = Utils.eval(predict_label_list)
			print("score:",score)

			loss_test /= testSetGenerater.batch_num

			print("loss:",loss_test)
			score = Utils.eval5(predict5_label_list)
			print("predict 5 score:",score)
		print("*****DepartmentClassification.train*****: training successfully.")
	
	def predict(self, line):
		# 1、对待分类语句进行分词
		jieba_index_list = self.__getJiebaIndex(line)

		# 2、搜索待分类语句中的术语
		term_index_list = self.__getTermIndex(line)

		len_jieba = len(jieba_index_list)
		len_term = len(term_index_list)
		
		if len_jieba > 0:
			jieba_index_array = np.array(jieba_index_list)
		else:
			jieba_index_array = np.zeros([Config.MAX_LENGTH], dtype=np.int32)
			len_jieba = Config.MAX_LENGTH
			
		if len_term > 0:
			term_index_array = np.array(term_index_list)
		else:
			term_index_array = np.zeros([Config.MAX_LENGTH], dtype=np.int32)
			len_term = Config.MAX_LENGTH
			
		batch_size = 1

		param_feed = {
			self.model.input_q: [jieba_index_array] * batch_size,
			self.model.input_a: [term_index_array] * batch_size,
			self.model.seq_len_q: [len_jieba] * batch_size,
			self.model.seq_len_a: [len_term] * batch_size,
			self.model.batch_seq_len_q: len_jieba,
			self.model.batch_seq_len_a: len_term, 
			self.model.dropout_keep_prob: 1.0
		}
		
		predict_1 = self.sess.run(self.model.predict_top, feed_dict = param_feed)
		index = predict_1[1]
		department = [self.dep1_index_dict[str(idx)] for idx in index[0]]

		return department

	def __getJiebaIndex(self, line):
		words = []
		segments = pseg.cut(line)
		for word in segments:
			if word.flag not in ['x','l','d','ul']:
				words.append(word.word)

		word_idex_list = []
		for word in words:
			if word in self.jieba_word_dict:
				word_idex_list.append(self.jieba_word_dict[word])
		return word_idex_list
		
	def __getTermIndex(self, line):
		termSearch = TermsSearch()
		words = termSearch.do(line)
		
		word_idex_list = list()

		for word in words:
			if word in self.term_word_dict:
				word_idex_list.append(self.term_word_dict[word])

		word_idex_list.reverse()
		return word_idex_list
		
	def check_restore_parameters(self):
		ckpt = tf.train.get_checkpoint_state(os.path.dirname(Config.model_dir + 'checkpoint'))
		if ckpt and ckpt.model_checkpoint_path:
			print("Loading parameters for the keshifenlei from:"+ckpt.model_checkpoint_path)
			self.model.saver.restore(self.sess, ckpt.model_checkpoint_path)
		else:
			print("load parameters error!")
