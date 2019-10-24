import time
import ujson
import numpy as np

import jieba
import jieba.posseg as pseg
# jieba.enable_parallel(4) # 结巴并行模式，仅支持linux

from Config import Config

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

	@staticmethod
	def saveAsTxt(fileName, data):
		'''
		将内容保存为txt
		'''
		print('save .. ' + fileName)
		fp = open(fileName, "w", encoding = "utf-8")
		fp.write(str(data))
		fp.close()
		print("save data done")

	@staticmethod
	def splitByJieba(sentence):
		'''单线程'''
		line = list()
		words = pseg.cut(sentence)

		# 将句子分词，提取其中的 n-名词，v-动词，vn-动名词，f-方位词
		# 提取单个语气词

		for key in words:
			if (len(key.word)>1 and key.flag in ['n','v','vn','f']) or (len(key.word)==1 and key.flag in ['y']):
				line.append(key.word)
		return ' '.join(line)

	@staticmethod
	def splitByJieba_P(sentence):
		'''多线程'''
		line = list()
		words = pseg.dt.cut(sentence)

		# 将句子分词，提取其中的 n-名词，v-动词，vn-动名词，f-方位词
		# 提取单个语气词

		for key in words:
			if (len(key.word)>1 and key.flag in ['n','v','vn','f']) or (len(key.word)==1 and key.flag in ['y']):
				line.append(key.word)
		return ' '.join(line)

	@staticmethod
	def showRate(current, step, total):
		timeStr = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) 
		if current >= total:
			print("current rate: 100% " + timeStr)
		elif current % step == 0:
			print("current rate: " + str(round(float(current * 100) / total, 2)) + "% " + timeStr)

	@staticmethod
	def eval(predict_label_and_marked_label_list):
		right_label_num = 0  #
		all_marked_label_num = 0	#label num
		for predict_labels, marked_labels in predict_label_and_marked_label_list:
			all_marked_label_num += 1
			if predict_labels == marked_labels:	 #point
				right_label_num += 1
		
		precision = float(right_label_num) / all_marked_label_num

		return precision
	
	@staticmethod
	def eval5(predict_label_and_marked_label_list):
		right_label_num = 0  #
		all_marked_label_num = 0	#label num
		for predict_labels, marked_labels in predict_label_and_marked_label_list:
			all_marked_label_num += 1
			if marked_labels in predict_labels:	 #point
				right_label_num += 1
		precision = float(right_label_num) / all_marked_label_num
		return precision