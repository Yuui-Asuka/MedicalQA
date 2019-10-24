import numpy as np

from Utils import Utils
from Config import Config

class DataGenerater():
	'''构造训练数据，以医学术语、jieba分词作为x，科室分类作为y'''
	def __init__(self, questionIndexList, batch_size = 128, training = True):
		# 1、初始化成员变量
		self.__init_property(training, batch_size)

		# 2、构造新的问题索引列表
		newQuestionIndexList = self.__buildNewQuestionList(questionIndexList)
		
		# 3、将数据拆分成若干批次
		self.__buildBatches(newQuestionIndexList)

	def __init_property(self, training, batch_size):
		print("*****DataGenerater.__init_property*****: init property begin...")
		self.training = training
		self.batch_size = batch_size	# 每个批次数据量
		self.batch_num = None   # 批次总数，后续计算
		self.batch_indexes = None # 批次索引列表

		self.all_jieba = list()
		self.all_jieba_len = list()
		self.all_term = list()
		self.all_term_len = list()
		self.all_dep1 = list()

		# cursor will be the cursor for the ith bucket
		self.cursor = 0
		self.epochs = 0

		print("*****DataGenerater.__init_property*****: init property successfully")

	def __buildNewQuestionList(self, questionIndexList):
		'''
		@result newQuestionIndexList:
			结构：
			[
				[len(jieba), jieba, len(term), term, dep1],
				...
			]
			示例：
			[
				(
					7, 
					['210', '285', '286', '5', '287', '35', '100'], 
					44, 
					['1454', '1455', '534', '73', '1456', '17', '25', '71', '72', '920', '78', '775', '1093', '606', '1457', '17', '914', '292', '131', '1458', '6', '168', '25', '1417', '78', '346', '74', '534', '198', '1456', '369', '1454', '21', '237', '1459', '6', '208', '534', '74', '124', '6', '1460', '509', '120'], 
					'4'
				),
				...
			]
		'''
		print("*****DataGenerater.__buildNewQuestionList*****: build len(jieba), jieba, len(term), term, dep1 list begin...")
		newQuestionIndexList = list()

		for q in questionIndexList:
			newQuestionIndexList.append((len(q[3]), q[3], len(q[2]), q[2], q[0]))

		# 排序的目的是什么？
		if self.training:
			newQuestionIndexList.sort(reverse=True)#sort by len desc
		#data_list.sort()

		Utils.saveAsTxt("./test/DataGenerater.__buildNewQuestionList.newQuestionIndexList.txt", newQuestionIndexList)
		print("*****DataGenerater.__buildNewQuestionList*****: build successfully.")
		return newQuestionIndexList

	def __buildBatches(self, newQuestionIndexList):
		# 计算可拆分的批次总数
		print("*****DataGenerater.__buildBatches*****: build batch begin...")
		self.batch_num = int((len(newQuestionIndexList)-1) / self.batch_size) + 1

		print("self.batch_num: " + str(self.batch_num))

		# 生成批次索引列表
		self.batch_indexes = [i for i in range(self.batch_num)]
		Utils.saveAsTxt("./test/DataGenerater.__buildBatches.batch_indexes.txt", str(self.batch_indexes))

		total = len(newQuestionIndexList)

		for index in self.batch_indexes:
			beginIndex = index * self.batch_size
			endIndex = min((index + 1) * self.batch_size, total)
			thisBatch = newQuestionIndexList[beginIndex:endIndex]

			tb_jieba = list()
			tb_jieba_len = list()
			tb_term = list()
			tb_term_len = list()
			tb_dep1 = list()

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

		Utils.saveAsTxt("./test/DataGenerater.__buildBatches.all_jieba_len.txt", str(self.all_jieba_len))
		Utils.saveAsTxt("./test/DataGenerater.__buildBatches.all_jieba.txt", str(self.all_jieba))
		Utils.saveAsTxt("./test/DataGenerater.__buildBatches.all_term_len.txt", str(self.all_term_len))
		Utils.saveAsTxt("./test/DataGenerater.__buildBatches.all_term.txt", str(self.all_term))
		Utils.saveAsTxt("./test/DataGenerater.__buildBatches.all_dep1.txt", str(self.all_dep1))

		print("*****DataGenerater.__buildBatches*****: build batch successfully.")

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
		print("*****DataGenerater.__getNextBatch*****: get current batch begin...")
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

		Utils.saveAsTxt("./test/DataGenerater.__getNextBatch.%s_%s_jieba_len.txt" % (str(self.epochs), str(self.cursor)), str(tb_jieba_len))
		Utils.saveAsTxt("./test/DataGenerater.__getNextBatch.%s_%s_jieba.txt" % (str(self.epochs), str(self.cursor)), str(tb_jieba))
		Utils.saveAsTxt("./test/DataGenerater.__getNextBatch.%s_%s_term_len.txt" % (str(self.epochs), str(self.cursor)), str(tb_term_len))
		Utils.saveAsTxt("./test/DataGenerater.__getNextBatch.%s_%s_term.txt" % (str(self.epochs), str(self.cursor)), str(tb_term))
		Utils.saveAsTxt("./test/DataGenerater.__getNextBatch.%s_%s_dep1.txt" % (str(self.epochs), str(self.cursor)), str(tb_dep1))

		print("*****DataGenerater.__getNextBatch*****: get current batch successfully.")
		return tb_jieba, tb_jieba_len, tb_term, tb_term_len, tb_dep1

	def __buildMatrix(self, keywords, keyword_len):
		'''生成一个固定长宽的训练集矩阵'''
		print("*****DataGenerater.__buildMatrix*****: build matrix begin")
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

		print("*****DataGenerater.__buildMatrix*****: build matrix successfully")

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
		print("debug caowei")
		'''
		注意，下列返回值中分词和术语的第二维维度不一定，是根据实际数据不同
		@result list() x_jieba
			[
				[  15  122   58   15]
 				[ 145  315  180  180]
 				...
 				[  98   98 1061    0]
 				[  98  200   98    0]
 			]
		@result list() x_jieba_len
			[4, 4, ..., 3, 3]

		@result list() x_term
			[
				[2032, 20, 2136, 1019, 2137, 2138, 2139, 2140, 1352, 394, 
				2032, 215, 1352, 1419, 2141, 28, 2142, 2143, 477, 31, 
				337, 1639, 0, 0, 0, 0, 0, 0, 0, 0, 
				0, 0, 0, 0, 0, 0, 0, 0, 0],

				[156, 1460, 1877, 325, 326, 16, 706, 16, 197, 396, 
				1771, 16, 1555, 5, 24, 3340, 16, 316, 783, 361, 
				3805, 1046, 420, 982, 971, 16, 3215, 1002, 449, 16, 
				9, 375, 0, 0, 0, 0, 0, 0, 0]
			]
		@result list() x_term_len
			[22, 32, ...]

		@result list() y_dep1 科室的独热编码
			[
				[
					0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 
					0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
					0.0, 0.0, 0.0
				]
				[
					0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
					0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
					0.0, 0.0, 0.0
				]
			]

		@result int max_len_jieba jieba中数量的最大值
		@result int max_len_term 术语中数量的最大值
		'''
		# 1、获取当前一个批次数据索引
		tb_jieba, tb_jieba_len, tb_term, tb_term_len, tb_dep1 = self.__getNextBatch()

		# 2、生成分词矩阵
		x_jieba, x_jieba_len, max_len_jieba = self.__buildMatrix(tb_jieba, tb_jieba_len)
		Utils.saveAsTxt("./test/DataGenerater.next_batch.%s_%s_x_jieba.txt" % (str(self.epochs), str(self.cursor)), str(x_jieba))
		Utils.saveAsTxt("./test/DataGenerater.next_batch.%s_%s_x_jieba_len.txt" % (str(self.epochs), str(self.cursor)), str(x_jieba_len))
		print("max_len_jieba: " + str(max_len_jieba))

		# 3、生成术语矩阵
		x_term, x_term_len, max_len_term = self.__buildMatrix(tb_term, tb_term_len)
		Utils.saveAsTxt("./test/DataGenerater.next_batch.%s_%s_x_term.txt" % (str(self.epochs), str(self.cursor)), str(x_term.tolist()))
		Utils.saveAsTxt("./test/DataGenerater.next_batch.%s_%s_x_term_len.txt" % (str(self.epochs), str(self.cursor)), str(x_term_len))
		print("max_len_term: " + str(max_len_term))

		# 4、如果是训练，需要生成标签
		y_dep1 = self.__buildLabel(tb_dep1)	 # 存疑
		Utils.saveAsTxt("./test/DataGenerater.next_batch.%s_%s_y_dep1.txt" % (str(self.epochs), str(self.cursor)), str(y_dep1.tolist()))
		
		return x_jieba, x_term, y_dep1, x_jieba_len, x_term_len, max_len_jieba, max_len_term