from Utils import Utils
from Config import Config
from DataGenerater import DataGenerater
from DataPrepared import DataPrepared
from DepartmentClassification import DepartmentClassification
from AnswerDataPrepared import AnswerDataPrepared
from MatchQuestion import MatchQuestion

# 原始数据集
originalData = "./qas_2K.csv"
#originalData = "./qas_1k_id_t.csv"
#originalData = "./qas_30W.csv"
# originalData = "./qas_30W_id_t.csv"

 #1、问题和回答数据预处理（一次处理，形成缓存）
#import time
#start =time.clock()
#AnswerDataPrepared(originalData)
#end = time.clock()
#print('Running time: %s Seconds'%(end-start))

 #2、科室分类模型数据预处理
#import time
#start = time.clock()
#dc = DataPrepared()
#dc.prepare(originalData, 300000)
#end = time.clock()
#print('Running time: %s Seconds'%(end-start))

# 3、构造科室分类数据生成器
#trainSet = Utils.readTrainData(Config.path_trainSet)
#testSet = Utils.readTrainData(Config.path_testSet)

#trainSetGenerater = DataGenerater(trainSet, Config.batch_size)
#testSetGenerater = DataGenerater(testSet, Config.batch_size)

##4、训练科室分类模型，构造科室分类器
dc = DepartmentClassification()
#dc.init_model(model_load = False)
#dc.train()

dc.init_model()
department = dc.predict("急慢性肾炎怎么预防")
print(department)

# 5、分类问题，匹配结果
# mq = MatchQuestion()
# department = "五官科"
# sentence = "外耳道炎三天了，现在特别疼，正常吗？多久能好"

# mq.match1(department, sentence)
# mq.match2(department, sentence)