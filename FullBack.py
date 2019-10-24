# -*- coding: utf-8 -*-

import time,datetime,copy,re,math
from pprint import pprint

from decimal import *
import csv
import pandas as pd
import numpy as np
# import ujson
# import config 
from sklearn.cross_validation import train_test_split
import sys
import ujson
import importlib
importlib.reload(sys)
import jieba
import jieba.posseg as pseg

class TermsSearch:
	'''
	搜索句子中存在的关键词
	以关键词的最后一个字在句子中搜索，找到该关键字之后，反向向前匹配关键词
	'''
	def __init__(self, encode='utf-8'):
		self.terms = self.prepare_terms()
		self.words = self.prepare_words(self.terms)

	def do(self, text):
		return self.search(self.terms, self.words, text)

	def prepare_terms(self):
		'''
		读取所有症状、部位、药品等关键词，形成一个字典
		terms = {'word': 1, "word2": 2, ...}
		'''
		terms = dict()
		index = 1
		basepath = './symptom/'
		fileNames = ['symptom.txt','imple.txt','yixuedaquan1.txt','yixuedaquan2.txt','disease.txt',"buwei.txt","xunyiwenyao.txt"]
		# fileNames = ['symptom.txt']
		for file in fileNames: 
			file =  open(basepath+file, 'r', encoding='utf-8')# as file:#py3
			
			temp_lines = file.readlines()

			for line in temp_lines:
				line = line.strip("\n").strip("\r").strip()

				if line and (line not in terms):
					terms[line]=index
					index += 1
		return terms

	def prepare_words(self, terms):
		'''
		获取所有词语的最后一个字，形成一个字典：
		wordDic = {"w1": 1, "w2": 2}
		'''
		words = set()
		
		for word in terms:
			words.add(word[len(word) - 1])

		wordDic = dict()
		index = 1

		for word in words:
			wordDic[word] = index
			index += 1

		return wordDic

	def search(self, terms, words, text):
		'''
		在一段文本中搜索关键词

		'''
		scope = 10 # 关键词搜索范围
		index = length = len(text) - 1
		recognised = list() # 最终匹配到的关键词

		while index > 0:
			word = text[index]
			lastIndex = None # 最长匹配到关键词的位置
			matches = list() # 搜索到的关键词

			if word in words:
				for point in range(1, min(scope, index)):
					if text[index-point:index] in terms:
						lastIndex = point
						matches.append(text[index-point:index])

				if lastIndex != None:
					index = index - lastIndex
					recognised.append(matches[-1]) # 为什么只要最后一个词？
				else:
					index = index - 1
			else:
				index = index - 1
		return recognised

def test_TermsSearch():
	terms = read_lines()
	print("terms is done")

	wordDic = set_dta(terms)
	print("wordDic is done")

	text = "'你好邓大夫，我妈妈骑电车意外碰','2015-08-13 22:36:18','你好邓大夫，我妈妈骑电车意外碰头上，就是眉毛上位置！！现在很疼！！有什么药可以缓解她现在的疼痛！','你好，伤口给予清创换药，疼的厉害可以口服头孢类消炎药及止疼药物。'"

	res = cut2(terms, wordDic, text)
	print(res)

	# 原始：['头', '口', '口', '现在', '现在', '眉毛', '头']
	# 改版：['头', '口', '口', '现在', '眉毛', '头']

class Config:
	"""docstring for Config"""
	
	batch_size = 64

	MAX_LENGTH = 128
	MAX_LABELS = 23	 # 

	NUM_EPOCHS = 10 # 训练轮数

	path_dep1_word_dict = "./train/dep1_word_dict.ujson"
	path_dep1_index_dict = "./train/dep1_index_dict.ujson"

	path_dep2_word_dict = "./train/dep2_word_dict.ujson"
	path_dep2_index_dict = "./train/dep2_index_dict.ujson"

	path_term_word_dict = "./train/term_word_dict.ujson"
	path_term_index_dict = "./train/term_index_dict.ujson"

	path_jieba_word_dict = "./train/jieba_word_dict.ujson"
	path_jieba_index_dict = "./train/jieba_index_dict.ujson"

	path_trainSet = "./train/trainSet.txt"
	path_testSet = "./train/testSet.txt"

	# ================= old ===========================
	
	valid_data_path = "../data/valid_data_set.csv"
	word_embedding_dict_file="word_embedding_test"


	model_dir="./models/"
	data_dir="./"
	model_path = "./models/keshi_fenlei_bilstm"
	pre_data_path = "../data/valid_homepage_prob.csv"

	NUM_LAYERS=1

	max_gradient_norm=5.0

	pre_batch_size = 1

	learning_rate=0.001

	learning_rate_decay_factor=0.9

	test_rate=0.1

	keep_prob = 0.6

	steps_per_checkpoint=8

	MAX_TRAIN_SAMPLE_NUM = 1000000
	MAX_RAW_SAMPLE_NUM = 1000

	EMBEDDING_DIM = 256
	
	ATTENTION_SIZE = 128
	HIDDEN_SIZE = 256
	FULL_HIDDEN_SIZE = 512
	
	

	

	num_checkpoints=50
	vocabulary_sw=0
	

	init_embedding=True

class Utils:
	@staticmethod
	def save_to_ujson(fileName, data):
		print('save .. ' + fileName)
		fp = open(fileName,"w")
		ujson.dump(data, fp)
		fp.close()
		print("save data done")

	@staticmethod
	def read_from_ujson(fileName):
		print('load .. ' + fileName)
		fp = open(fileName,"rb")
		data =  ujson.load(fp)
		fp.close()
		print("load data done")
		return data

	@staticmethod
	def saveTrainData(fileName, data):
		'''
		将问题数据拼接成字符串保存到文件，格式为：
		dep_1;dep_2;term1,term2,term3;jieba1,jieba2,jieba3\n
		'''
		print('save .. ' + fileName)

		template = "%s;%s;%s;%s\n"
		fp = open(fileName,"w")

		for q in data:
			termStr = ",".join(str(word) for word in q[2])
			jiebaStr = ",".join(str(word) for word in q[3])
			row = "%s;%s;%s;%s\n" % (str(q[0]), str(q[1]), termStr, jiebaStr)
			fp.write(row)
			fp.flush()

		fp.close()
		print("save data done")

	@staticmethod
	def readTrainData(fileName):
		print('read .. ' + fileName)
		fp = open(fileName,"r")

		questionIndexList = list()
		
		while True:
			q = list()
			line = fp.readline()
			if not line:
				break
			line = line.strip("\n").strip("\r").split(";")
			q.append(line[0])
			q.append(line[1])
			
			if len(line[2]) > 0:
				q.append(line[2].split(","))
			else:
				q.append([0])

			if len(line[3]) > 0:
				q.append(line[3].split(","))
			else:
				q.append([0])
			
			questionIndexList.append(q)
		fp.close()
		print("read data done")
		return questionIndexList

	@staticmethod
	def get_y_lable(temp):
		label = np.zeros(Config.MAX_LABELS)
		label[int(temp)] = 1
		return label

class DataPrepared:
	def prepare(self):
		# 1、数据预处理
		questionList, statistic_dep_1, statistic_dep_2, statistic_term, statistic_jieba = self.__preproccess("./qas.csv", 10000)
		
		# 2、将部门、术语、分词生成索引字典，并保存到文件
		dep1_word_dict, dep1_index_dict = self.__generatePairDict(statistic_dep_1, 0)
		Utils.save_to_ujson(Config.path_dep1_word_dict, dep1_word_dict)
		Utils.save_to_ujson(Config.path_dep1_index_dict, dep1_index_dict)

		dep2_word_dict, dep2_index_dict = self.__generatePairDict(statistic_dep_2, 0)
		Utils.save_to_ujson(Config.path_dep2_word_dict, dep2_word_dict)
		Utils.save_to_ujson(Config.path_dep2_index_dict, dep2_index_dict)

		# 为什么这里索引从1开始
		term_word_dict, term_index_dict = self.__generatePairDict(statistic_term, 1)
		Utils.save_to_ujson(Config.path_term_word_dict, term_word_dict)
		Utils.save_to_ujson(Config.path_term_index_dict, term_index_dict)

		jieba_word_dict, jieba_index_dict = self.__generatePairDict(statistic_jieba, 1)
		Utils.save_to_ujson(Config.path_jieba_word_dict, jieba_word_dict)
		Utils.save_to_ujson(Config.path_jieba_index_dict, jieba_index_dict)
		
		# 3、生成问题和部门、术语、分词之间的索引列表
		questionIndexList = self.__generateQuestionIndex(questionList, dep1_word_dict, 
			dep2_word_dict, term_word_dict, jieba_word_dict)
		
		# 4、拆分训练集和测试集
		train_set, test_set = train_test_split(questionIndexList,test_size=0.1)

		# 5、将训练集和测试集保存到文件
		Utils.saveTrainData(Config.path_trainSet, train_set)
		Utils.saveTrainData(Config.path_testSet, test_set)

	def __generatePairDict(self, statistic, start):
		word_dict = dict()
		index_dict = dict()

		for key in statistic.keys():
			word_dict[key] = start
			index_dict[start] = key
			start += 1

		return word_dict, index_dict

	def __generateQuestionIndex(self, questionList, dep1_word_dict, dep2_word_dict, 
		term_word_dict, jieba_word_dict):
		'''
		生成问题到科室、术语、分词的索引。结构：
		[
			[
				dep_1_index, 
				dep_2_index, 
				[split_1_index, split_2_index, ...],
				[term_1_index, term_2_index, ...]
			]
		]
		'''
		questionIndexList = list()
		
		for q in questionList:
			dep_1 = dep1_word_dict[q[0]]
			dep_2 = dep2_word_dict[q[1]]

			term_index = list()
			for voc in q[2]:
				term_index.append(term_word_dict[voc])

			if len(term_index) == 0:
				term_index.append(0)

			jieba_index = list()
			for voc in q[3]:
				jieba_index.append(jieba_word_dict[voc])

			questionIndexList.append((dep_1, dep_2, term_index, jieba_index))
			
		return questionIndexList

	# 数据预处理 __preproccess 开始

	def __preproccess(self, fileName, MAX_HANDLE_NUM):
		'''MAX_HANDLE_NUM 最大处理数量'''

		csvReader = csv.reader(open(fileName, encoding="utf-8", errors="ignore")) 

		processecCount = 0
		existedQuestion = set() # 用于问题去重

		statistic_dep_1 = dict()
		statistic_dep_2 = dict()
		statistic_term_search_failed = 0
		statistic_term = dict()
		statistic_jieba = dict()

		questionList = list()

		for index, row in enumerate(csvReader):
			# 1、显示进度、检查是否超出最大处理范围
			self.__checkRangeAndShowRate(processecCount, MAX_HANDLE_NUM)
			processecCount += 1

			# 2、数据完整性检查
			if not self.__checkIntegrality(row):
				continue

			# 3、取出科室、问题描述数据（如有编码问题，需处理）
			dep_1, dep_2, question, detail = self.__checkEncodingAndExtractQues(row)
			if dep_1 ==None or dep_2 == None or question == None or detail == None:
				continue

			# 4、问题去重，按照问题描述去重
			if detail in existedQuestion:
				continue			
			existedQuestion.add(detail)

			# 5、过滤指定科室数据不处理，是不是特定针对不同来源的数据进行不同处理？
			if (filter == True) and (level1 in filter_keshi_list):
				continue

			# 6、统计科室下问题数量
			self.__statisticsDepartmentQues(dep_1, dep_2, statistic_dep_1, statistic_dep_2)

			# 7、搜索医学术语，使用jieba分词
			vocabulary_term, vocabulary_jieba = self.__termSearchAndSplit(question, detail)
			if len(vocabulary_jieba) <= 0:
				continue
			
			# 8、统计未搜索到医学术语、以及各类术语、分词出现的情况
			statistic_term_search_failed = self.__statisticsTermAndJieba(vocabulary_term, vocabulary_jieba, 
				statistic_term_search_failed, statistic_term, statistic_jieba)

			questionList.append((dep_1, dep_2, vocabulary_term, vocabulary_jieba))

		return questionList, statistic_dep_1, statistic_dep_2, statistic_term, statistic_jieba

	def __checkIntegrality(self, row):
		'''
		检查数据完整性：True-完整；False-不完整
		1、是否为空数据，跳过空数据（明确数据源不存在非空数据，跳过该步骤）
		2、是否包含必要的科室、问题描述
		'''
		if len(row) >= 4:
			return True
		else:
			return False

	def __checkRangeAndShowRate(self, processecCount, max):
		'''
		判断是否超出最大处理数量范围，以及显示处理进度
		'''
		# 1、显示处理进度
		if (processecCount * 10) % max == 0:
				print("current rate:" + str(float(processecCount * 100) / max) + "%")

		# 2、最大处理数量检查点
		if processecCount > max: #read only max_num row
			return True
		else:
			return False

	def __checkEncodingAndExtractQues(self, row):
		'''
		如果内容是采用非UTF-8编码，此处应进行转码
		"XXXXX".decode("gbk").encode("utf8")
		这里明确是UTF-8编码
		'''
		dep_1 = dep_2 = question = detail = None

		try:
			if row[0] == None or row[0] == "":
				dep_1 = None
			else:
				dep_1 = row[0] # .decode("gbk").encode("utf8")

			if row[1] == None or row[1] == "":
				dep_2 = row[1]
			else:
				dep_2 = dep_1 # 如果没有二级科室，则使用一级科室代替
			question = row[2]
			detail = row[3]		
		except:
			dep_1 = dep_2 = question = detail = None

		return dep_1, dep_2, question, detail

	def __statisticsDepartmentQues(self, dep_1, dep_2, statistic_1, statistic_2):
		'''统计一级、二级每个科室下的问题数量'''
		if dep_1 in statistic_1:
			statistic_1[dep_1] += 1
		else:
			statistic_1[dep_1] = 1

		if dep_2 in statistic_2:
			statistic_2[dep_2] += 1
		else:
			statistic_2[dep_2] = 1

	def __termSearchAndSplit(self, question, detail):
		'''使用TermsSearch定义的医学关键词进行搜索，使用jieba进行分词'''
		termSearch = TermsSearch()
		vocabulary_term = termSearch.do(detail)
		vocabulary_term.reverse()

		vocabulary_jieba = list()
		words = pseg.cut(detail)
		for word in words:
			if word.flag not in ['x','l','d','ul']:
				vocabulary_jieba.append(word.word)

		return vocabulary_term, vocabulary_jieba

	def __statisticsTermAndJieba(self, vocabulary_term, vocabulary_jieba, 
		statistic_term_search_failed, statistic_term, statistic_jieba):
		'''
		统计搜索医学术语失败的记录、医学术语频度、结巴分词频度
		失败记录是传值，所以返回；另外两个统计是传引用，不需要返回
		'''
		if len(vocabulary_term) <= 0:
			statistic_term_search_failed += 1

		for voc in vocabulary_term:
			if voc not in statistic_term:
				statistic_term[voc] = 0
			statistic_term[voc] += 1

		for voc in vocabulary_jieba:
			if voc not in statistic_jieba:
				statistic_jieba[voc] = 0
			statistic_jieba[voc] += 1

		return statistic_term_search_failed

	# 数据预处理 __preproccess 结束

def test_DataPrepared():
	dc = DataPrepared()
	dc.prepare()

class DataGenerater():
	'''构造训练数据，以医学术语、jieba分词作为x，科室分类作为y'''
	def __init_property(self, training, batch_size):
		self.training = training
		self.batch_size = batch_size	# 每个批次数据量
		self.batch_num = None   # 批次总数，后续计算
		self.batch_indexes = None # 批次索引列表

		self.all_jieba = self.all_jieba_len = list()
		self.all_term = self.all_term_len = list()
		self.all_dep1 = list()

		# cursor will be the cursor for the ith bucket
		self.cursor = 0
		self.epochs = 0

	def __buildNewQuestionList(self, questionIndexList):
		'''
		newList = [
			[len(jieba), jieba, len(term), term, dep1],
			...
		]
		'''
		newQuestionIndexList = list()

		for q in questionIndexList:
			newQuestionIndexList.append((len(q[2]), q[2], len(q[3]), q[3], q[0]))

		print(newQuestionIndexList[0])

		# 排序的目的是什么？
		if self.training:
			newQuestionIndexList.sort(reverse=True)#sort by len desc
		#data_list.sort()
		return newQuestionIndexList

	def __buildBatches(self, newQuestionIndexList):
		# 计算可拆分的批次总数
		self.batch_num = int((len(newQuestionIndexList)-1) / self.batch_size) + 1

		# 生成批次索引列表
		self.batch_indexes = [i for i in range(self.batch_num)]

		total = len(newQuestionIndexList)

		for index in self.batch_indexes:
			beginIndex = index * self.batch_size
			endIndex = min((index + 1) * self.batch_size, total)
			thisBatch = newQuestionIndexList[beginIndex:endIndex]

			tb_jieba = tb_jieba_len = tb_term = tb_term_len = tb_dep1 = list()

			for row in thisBatch:
				tb_jieba_len.append(row[0])
				tb_jieba.append(row[1])
				tb_term_len.append(row[2])
				tb_term.append(row[3])
				tb_dep1.append(row[4])

			self.all_jieba_len.append(tb_jieba_len)
			self.all_jieba.append(tb_jieba)
			self.all_term_len.append(tb_term_len)
			self.all_term.append(tb_term)
			self.all_dep1.append(tb_dep1)

		if self.training:
			self.shuffle()

	def __init__(self, questionIndexList, batch_size = 128, training = True):
		# 1、初始化成员变量
		self.__init_property(training, batch_size)

		# 2、构造新的问题索引列表
		newQuestionIndexList = self.__buildNewQuestionList(questionIndexList)
		
		# 3、将数据拆分成若干批次
		self.__buildBatches(newQuestionIndexList)

	def shuffle(self):
		self.cursor = 0
		np.random.shuffle(self.batch_indexes) # index shuffle need fix

	def shuffle_data(self, data):
		return np.random.permutation(data)

	def dropout_data(self, data, p=0.5):
		data_len = len(data)
		select_len = int( np.ceil(data_len * p) )
		data_p = random.sample(data,select_len)
		return data_p, select_len 

	def __getNextBatch(self):
		'''一整批处理完后，混淆数据顺序，重新开始获取'''
		if self.cursor >= self.batch_num:
			self.epochs += 1
			if self.training:
				self.shuffle()

		tb_Index = self.batch_indexes[self.cursor]
		self.cursor += 1
		
		tb_jieba = self.all_jieba[tb_Index]
		tb_term = self.all_term[tb_Index]
		tb_dep1 = self.all_dep1[tb_Index]
		
		tb_jieba_len = self.all_jieba_len[tb_Index]
		tb_term_len = self.all_term_len[tb_Index]

		return tb_jieba, tb_jieba_len, tb_term, tb_term_len, tb_dep1

	def __buildMatrix(self, keywords, keyword_len):
		'''生成一个固定长宽的训练集矩阵'''
		max_len = max(keyword_len)

		if max_len > Config.MAX_LENGTH:
			max_len = Config.MAX_LENGTH

		tb_size = len(keywords)

		# 生成一个 tb_size * max_len 的0填充的矩阵
		x = np.zeros([tb_size, max_len], dtype = np.int32)

		for i, x_i in enumerate(x):
			if keyword_len[i] > max_len:
				len_temp = max_len
				keyword_len[i] = max_len
			else:
				len_temp = keyword_len[i]
		 
			x_i[:len_temp] = keywords[i][:len_temp]

		return x, keyword_len, max_len

	def __buildLabel(self, tb_dep1):
		if self.training:
			y = []
			for temp in tb_dep1: #each sample
				temp_label = Utils.get_y_lable(temp)#temp is a list of one sample
				y.append(temp_label)
			y = np.array(y, dtype = np.float32)
		else:
			y = tb_dep1
		return y

	def next_batch(self):

		# 1、获取当前一个批次数据索引
		tb_jieba, tb_jieba_len, tb_term, tb_term_len, tb_dep1 = self.__getNextBatch()
		print(tb_jieba)
		print(tb_jieba_len)
		# 2、生成分词矩阵
		x_jieba, x_jieba_len, max_len_jieba = self.__buildMatrix(tb_jieba, tb_jieba_len)

		# 3、生成术语矩阵
		x_term, x_term_len, max_len_term = self.__buildMatrix(tb_term, tb_term_len)
  
		# 4、如果是训练，需要生成标签
		y_dep1 = self.__buildLabel(tb_dep1)	 # 存疑
		
		return x_jieba, x_term, y_dep1, x_jieba_len, x_term_len, max_len_jieba, max_len_term

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
			loss_tr, _, prey = self.sess.run(fetches, feed_dict=param_feed)
			
			#print(prey)

		print("loss_train:",loss_train)
		path = self.model.saver.save(self.sess, Config.model_path, global_step=i)
		print("Saved model checkpoint to {}".format(path))

		return loss_tr

	def __train_test(set, testSetGenerater):
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
		# 1、获取训练和测试数据
		trainSet = Utils.readTrainData(Config.path_trainSet)
		testSet = Utils.readTrainData(Config.path_testSet)

		# 2、创建训练、测试数据生成器
		trainSetGenerater = DataGenerater(trainSet, Config.batch_size)
		testSetGenerater = DataGenerater(testSet, Config.batch_size)

		# 3、训练和测试模型
		for epoch in range(Config.NUM_EPOCHS):

			# 3.1、训练模型
			loss_train = 0
			loss_tr = self.__train_train(trainSetGenerater)
			loss_train = loss_tr * self.DELTA + loss_train * (1 - self.DELTA)

			# 3.2、测试模型
			loss_test = 0
			predict_label_list, predict5_label_list, loss_test_batch = self.__train_test(testSetGenerater)
			loss_test += loss_test_batch

			score = data_utils.eval(predict_label_list)
			print("score:",score)

			loss_test /= testSetGenerater.batch_num

			print("loss:",loss_test)
			score = data_utils.eval5(predict5_label_list)
			print("predict 5 score:",score)
	
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

def test_DataGenerater():
	trainSet = Utils.readTrainData(Config.path_trainSet)
	testSet = Utils.readTrainData(Config.path_testSet)

	trainSetGenerater = DataGenerater(trainSet, Config.batch_size)
	testSetGenerater = DataGenerater(testSet, Config.batch_size)


	x_jieba, x_term, y_dep1, x_jieba_len, x_term_len, max_len_jieba, max_len_term = trainSetGenerater.next_batch()

	x_jieba, x_term, y_dep1, x_jieba_len, x_term_len, max_len_jieba, max_len_term = testSetGenerater.next_batch()

class ClassificationModel(object):
	def __init__(self,jiba_word_num,keyword_num):
		
		self.input_q = tf.placeholder(tf.int32, [None, None], name="input_q")#None
		self.input_a = tf.placeholder(tf.int32, [None, None], name="input_a")#None
		self.input_y = tf.placeholder(tf.float32, [None, config.MAX_LABELS], name="input_y")

		self.seq_len_q = tf.placeholder(tf.int32, [None], name="seq_len_q")
		self.seq_len_a = tf.placeholder(tf.int32, [None], name="seq_len_a")
		self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
		self.batch_seq_len_q = tf.placeholder(tf.int32, name="batch_seq_len_q")
		self.batch_seq_len_a = tf.placeholder(tf.int32, name="batch_seq_len_a")
		embeddings_weights_q = tf.get_variable('embeddings_weights_q', shape=[jiba_word_num, config.EMBEDDING_DIM], initializer=tf.uniform_unit_scaling_initializer())
		rnn_input_q = tf.nn.embedding_lookup(embeddings_weights_q, self.input_q)
		
		embeddings_weights_a = tf.get_variable('embeddings_weights_a', shape=[keyword_num, config.EMBEDDING_DIM], initializer=tf.uniform_unit_scaling_initializer())
		rnn_input_a = tf.nn.embedding_lookup(embeddings_weights_a, self.input_a)
		
		#Bilstm layer(-s)
		dynamic_rnn_outputs_q = rnn_input_q
		for i in range(config.NUM_LAYERS):
			with tf.variable_scope("bidirectional_rnn_q_%d" % i):
				(output, state) = tf.nn.bidirectional_dynamic_rnn(LSTMCell(config.HIDDEN_SIZE), LSTMCell(config.HIDDEN_SIZE),inputs=dynamic_rnn_outputs_q, sequence_length=self.seq_len_q, dtype=tf.float32)
				dynamic_rnn_outputs_q = tf.concat(output, 2)
				if i < config.NUM_LAYERS-1:
					print("add dropout between lstm")
					dynamic_rnn_outputs_q = tf.nn.dropout(dynamic_rnn_outputs_q, self.dropout_keep_prob)

		# attention layer
		attention_output_q = dynamic_attention(dynamic_rnn_outputs_q, config.ATTENTION_SIZE, self.batch_seq_len_q)
		
		#Bilstm layer(-s)
		dynamic_rnn_outputs_a = rnn_input_a
		for i in range(config.NUM_LAYERS):
			with tf.variable_scope("bidirectional_rnn_a_%d" % i):
				(output, state) = tf.nn.bidirectional_dynamic_rnn(LSTMCell(config.HIDDEN_SIZE), LSTMCell(config.HIDDEN_SIZE),inputs=dynamic_rnn_outputs_a, sequence_length=self.seq_len_a, dtype=tf.float32)
				dynamic_rnn_outputs_a = tf.concat(output, 2)
				if i < config.NUM_LAYERS-1:
					print("add dropout between lstm")
					dynamic_rnn_outputs_a = tf.nn.dropout(dynamic_rnn_outputs_a, self.dropout_keep_prob)

		# attention layer
		attention_output_a = dynamic_attention(dynamic_rnn_outputs_a, config.ATTENTION_SIZE, self.batch_seq_len_a)
		attention_output = tf.concat([attention_output_q,attention_output_a],1)
		# Dropout
		drop_outputs = tf.nn.dropout(attention_output, self.dropout_keep_prob)
		
		print("fc layer intput dim:"+str(drop_outputs.get_shape()[1].value))

		drop_outputs_full = drop_outputs
		# fully connected layer
		with tf.name_scope("output_layer"):
			W = tf.get_variable(
				"output_weight",
				shape=[drop_outputs_full.get_shape()[1].value, config.MAX_LABELS],
				initializer=tf.contrib.layers.xavier_initializer())
			b = tf.Variable(tf.constant(0.1, shape=[config.MAX_LABELS]), name="output_b")
			logits = tf.nn.xw_plus_b(drop_outputs_full, W, b, name="y_matmul")
			self.pre_y = tf.nn.softmax(logits,name="pre_y")
			#self.pre_y = tf.argmax(logits_temp, 1)
		# define loss
		with tf.name_scope("loss"):
			self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.input_y))

		# define optimizer
		with tf.name_scope("optimizer"):
			self.optimizer = tf.train.AdamOptimizer(config.learning_rate).minimize(self.loss)

		# define saver，only save last 5 models
		#saver = tf.train.Saver(tf.global_variables())
		self.saver = tf.train.Saver(tf.global_variables())
		self.predict_top = tf.nn.top_k(self.pre_y, k=5)
		self.predict_top1 = tf.nn.top_k(self.pre_y, k=1)
		self.label = tf.nn.top_k(self.input_y, k=1) 

test_DataGenerater()