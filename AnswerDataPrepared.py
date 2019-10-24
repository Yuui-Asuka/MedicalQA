import os
import pickle
import pandas as pd
import jieba.posseg as pseg

from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

from Utils import Utils
from Config import Config

class AnswerDataPrepared(object):
	"""docstring for AnswerDataPrepared"""
	def __init__(self, fileName):
		# 1、构造部门与问题分词的映射
		self.buildDepartQuesMap2(fileName)
		self.buildIdAnsMap(fileName)

	def buildDepartQuesMap(self, fileName):
		data = pd.read_csv(fileName, encoding = "utf-8")

		# 对数据去重，抓取数据的标题是问题描述的一部分，这里抛弃标题
		# qid,department_1,department_2,question,detail,treatment,demand,gender,age,time,analysis,advice
		data = data[["qid", "department_1", "detail"]].drop_duplicates()
		data.reset_index(inplace=True, drop=True)

		for i in tqdm(data.index):
			data.loc[i, 'detail'] = data["detail"][i].replace("\r", "").replace("\n", "")
			data.loc[i,'detail_split'] = Utils.splitByJieba(data["detail"][i])
	
		pickle.dump(data, open(Config.path_cache_ques_all, 'wb+'))
		Utils.saveAsTxt("./test/AnswerDataPrepared.buildDepartQuesMap.questionSplit.txt", str(data.sample(10)))

		for department in set(data["department_1"]):
			filterQuestion = data[data["department_1"] == department]
			filterQuestionSplit = filterQuestion[["qid", "detail_split"]]
			filterQuestionSplit.reset_index(inplace=True, drop=True)
			pickle.dump(filterQuestionSplit, open('./cache/%s' % (department), 'wb+'))

	def buildDepartQuesMap2(self, fileName):
		dataframe = pd.read_csv(fileName, encoding = "utf-8")
		total = len(dataframe.index)

		departmentQuesDic = dict()
		repeatCheckDic = set()

		for index, row in dataframe.iterrows():
			Utils.showRate(index+1, 5000, total)
			if row["detail"] not in repeatCheckDic:
				repeatCheckDic.add(row["detail"])
			else:
				continue

			if row["department_1"] not in departmentQuesDic.keys():
				departmentQuesDic[row["department_1"]] = list()

			try:
				detail = row["detail"].replace("\r", "").replace("\n", "")
			except Exception as e:
				print(row["qid"])
				print(row["detail"])
				continue

			line = {
				"qid": row["qid"],
				"department_1" : row["department_1"],
				"detail": detail,
				"detail_split": Utils.splitByJieba(detail)

			}
			departmentQuesDic[row["department_1"]].append(line)

		for key in departmentQuesDic.keys():
			deptDataFrame = pd.DataFrame(departmentQuesDic[key])
			pickle.dump(deptDataFrame, open('./cache/%s' % (key), 'wb+'))
			Utils.saveAsTxt("./test/AnswerDataPrepared.buildDepartQuesMap2.depQues.%s.txt" % (key), str(departmentQuesDic[key]))


	def buildIdAnsMap(self, fileName):
		data = pd.read_csv(fileName, encoding = "utf-8")
		
		qIdAns = dict()

		for qid in set(data["qid"]):
			thQues = data[data["qid"] == qid]
			thQues.reset_index(inplace=True, drop=True)
			answer = str(thQues[["analysis"]]['analysis'][0]) + str(thQues[["advice"]]["advice"][0])
			qIdAns[qid] = answer
		
		pickle.dump(qIdAns, open(Config.path_cache_answer_all, 'wb+'))

		Utils.saveAsTxt("./test/AnswerDataPrepared.buildIdAnsMap.qidAndAnswerMap.txt", str(qIdAns))

		return qIdAns