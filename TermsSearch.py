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
