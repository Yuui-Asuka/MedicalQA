import pickle
import pandas as pd
from tqdm import tqdm
from Utils import Utils
from Config import Config
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

class MatchQuestion(object):
	def match1(self, department, sentence):
		dump_path = r'./cache/%s' % (department)
		dataframe = pickle.load(open(dump_path, 'rb+'))
		
		qidSmailityMap = self.__calculate_smailarity_1(sentence, dataframe)
		print(qidSmailityMap[0][0])
		for offset,i in enumerate(qidSmailityMap):
			index_que = dataframe[dataframe['qid'] == i[0]]
			index_que.reset_index(inplace=True, drop=True)

			print (offset,'最匹配问题：', index_que["detail_split"][0])
		
		print ('\n')
		print ('回答：',self.__recommend_rule(qidSmailityMap[0][0]))
		print ('\n')
		
		# print ('......way2......')
		# rs2 = cal_way2(input_sen, department, 'tf', 'cosVector')
		# for offset,i in enumerate(rs1):
		# 	iid = i[0]
		# 	print (offset,'最匹配问题：',list(df[df['ID00']==iid]['cut_WT00']))
		# 	print ('回答：',recommend_rule(iid))
	
	def match2(self, department, sentence):
		dump_path = r'./cache/%s' % (department)
		dataframe = pickle.load(open(dump_path, 'rb+'))
		
		qidSmailityMap = self.__calculate_smailarity_2(sentence, dataframe, 'tf', 'cosVector')
		print(qidSmailityMap[0][0])
		for offset,i in enumerate(qidSmailityMap):
			index_que = dataframe[dataframe['qid'] == i[0]]
			index_que.reset_index(inplace=True, drop=True)

			print (offset,'最匹配问题：', index_que["detail_split"][0])
		
		print ('\n')
		print ('回答：',self.__recommend_rule(qidSmailityMap[0][0]))
		print ('\n')


	def __calculate_smailarity_1(self, sentence, dataframe):
		# 集合法
		sentence = Utils.splitByJieba(sentence)
		words = sentence.split(' ')
		
		result = dict()

		for index in tqdm(dataframe.index):
			index_words = dataframe["detail_split"][index].split(' ')
			similarity = float(len(set(words) & set(index_words))) / len(set(words) | set(index_words))
			result[dataframe["qid"][index]] = similarity

		result = sorted(result.items(), key = lambda f:f[1], reverse=True)
		Utils.saveAsTxt("./test/MatchQuestion.__calculate_smailarity_1.smaility.txt", str(result))
		result = result[:3]
		return result

	def __calculate_smailarity_2(self, sentence, dataframe, tag1, tag2):
		# 距离法
		sentence = Utils.splitByJieba(sentence)
		words = sentence.split(' ')

		result = dict()
		for index in tqdm(dataframe.index):
			index_words = dataframe["detail_split"][index].split(' ')

			# 没有交集返回-1
			if len(set(words) & set(index_words)) == 0:
				similarity = -1
			else:
				quesList = [sentence, dataframe["detail_split"][index]]
				x = self.__calculate_tf_idf(quesList, tag1)
				if tag2 == 'Euclidean':
					try:
						similarity = float(self.__calculate_Euclidean(x.iloc[0].values, x.iloc[1].values))
					except:
						similarity = -1
				else:
					try:
						similarity = float(self.__calculate_Euclidean(x.iloc[0].values, x.iloc[1].values))
					except:
						similarity = -1
			result[dataframe['qid'][index]] = similarity
		result = sorted(result.items(), key=lambda f:f[1], reverse=True)
		Utils.saveAsTxt("./test/MatchQuestion.__calculate_smailarity_2.smaility.txt", str(result))
		result = result[:3]
		return result

	def __calculate_tf_idf(self, data_list, tag):
	    # 计算 TF 或者 TF-IDF值
	    vectorizer = CountVectorizer()
	    if tag == 'tf':
	        #获取TF矩阵（词频）
	        tf_matrix = vectorizer.fit_transform(data_list).toarray()
	        word = vectorizer.get_feature_names()
	        tf_matrix = pd.DataFrame(tf_matrix,columns=word)
	    else:
	        #获取TFIDF矩阵
	        transformer = TfidfTransformer()
	        tfidf = transformer.fit_transform(vectorizer.fit_transform(data_list))
	        word = vectorizer.get_feature_names()
	        weight = tfidf.toarray()
	        tf_matrix = pd.DataFrame(weight,columns=word)
	    idx_set = set(tf_matrix.columns)
	    tf_matrix = tf_matrix[list(idx_set)]
	    return tf_matrix

	def __calculate_Euclidean(p, q):
	    #计算欧几里德距离,并将其标准化
	    same = 0
	    for i in p:
	        if i in q:
	            same +=1
	    e = sum([(p[i] - q[i])**2 for i in range(same)])
	    return 1/(1+e**.5)

	def __recommend_rule(self, qid):
	    qidAnswerDic = pickle.load(open(Config.path_cache_answer_all, 'rb+'))
	    
	    if qid in qidAnswerDic.keys():
	    	return qidAnswerDic[qid]
	    else:
	    	return "UNKNOw"