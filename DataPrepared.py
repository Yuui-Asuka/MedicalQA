import csv
import jieba.posseg as pseg
from sklearn.cross_validation import train_test_split

from TermsSearch import TermsSearch
from Utils import Utils
from Config import Config

class DataPrepared:
	def prepare(self, fileName, maxHandleNum):
		
		# 1、数据预处理
		questionList, statistic_dep_1, statistic_dep_2, statistic_term, statistic_jieba = self.__preproccess(fileName, maxHandleNum)

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
			(
				1, 47, 
				[26, 229, 230, 131, 231, 232], 
				[1231, 143, 1232, 1233, 349, 29, 738, 1234, 466, 1235]
			), (
				2, 32, 
				[0], 
				[741, 1236, 444, 1237, 1238, 1239, 21, 174, 1240, 1241, 58, 14, 1242]
			), (
				4, 10, 
				[233, 233], 
				[1069, 25, 1243, 72, 1092, 135, 126, 1153, 942, 18, 173, 1244, 6, 477, 389, 1153, 942, 655, 1245, 583, 1088, 173, 1244, 6, 1156, 583, 395, 225, 375, 1246, 1247]
			)
		]
		'''
		print("*****DataPrepared.__generateQuestionIndex*****: generate dep1, dep2, [jieba], [term] list...")
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
		
		Utils.saveAsTxt("./test/DataPrepared.__generateQuestionIndex.questionIndexList.txt", str(questionIndexList))

		print("*****DataPrepared.__generateQuestionIndex*****: generate dep1, dep2, [jieba], [term] list...")
		return questionIndexList

	# 数据预处理 __preproccess 开始

	def __preproccess(self, fileName, MAX_HANDLE_NUM):
		'''
		MAX_HANDLE_NUM 最大处理数量
		
		@return list() questionList:
			[
				(
					'内科', '呼吸内科', 
					[], 
					['发烧', '有点', '晕', '忽冷忽热', '的', '38', '度', '三', '该']
				), (
					'内科', '神经内科', 
					['安眠', '安眠'], 
					['请问', '安眠药', '吃', '请问', '吃', '安眠药', '多少', '可以', '昏睡']
				), (
					'其他', '生活起居', 
					['5岁', '易饿'], 
					['35', '岁', '男性', '外表', '看着', '瘦', '抽烟', '晚睡', '前', '容易', '饿', '喜欢', '吃', '东西', '夜里', '有时', '醒来', '凌晨', '2', '点', '后', '睡', '凌晨', '爱', '盗汗', '需要', '怎样', '治疗', '呢', '谢谢']
				)
			]

		@return dict() statistic_dep_1: {'内科': 54, '外科': 31, '妇产科': 46	}
		@return dict() statistic_dep_2: {'呼吸内科': 8, '神经内科': 4, '外科其它': 1, '妇科综合': 8, '消化内科': 7}
		@return dict() statistic_term: {'安眠': 2, '5岁': 1, '安全期': 1, '月经': 15, '烦躁': 2, '眼': 7}
		@return dict() statistic_jieba: {'发烧': 4, '有点': 15, '晕': 2, '忽冷忽热': 1}
		'''
		print("*****DataPrepared.__preproccess*****: __preproccess data begin...")
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

		Utils.saveAsTxt("./test/DataPrepared.__preproccess.questionList.txt", str(questionList))
		Utils.saveAsTxt("./test/DataPrepared.__preproccess.statistic_dep_1.txt", str(statistic_dep_1))
		Utils.saveAsTxt("./test/DataPrepared.__preproccess.statistic_dep_2.txt", str(statistic_dep_2))
		Utils.saveAsTxt("./test/DataPrepared.__preproccess.statistic_term.txt", str(statistic_term))
		Utils.saveAsTxt("./test/DataPrepared.__preproccess.statistic_jieba.txt", str(statistic_jieba))

		print("*****DataPrepared.__preproccess*****: __preproccess data successfully.")

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
		Utils.showRate(processecCount, 1000, max)
		
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
				dep_2 = dep_1 # 如果没有二级科室，则使用一级科室代替
			else:
				dep_2 = row[1]
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