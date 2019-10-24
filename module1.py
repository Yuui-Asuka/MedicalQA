import time
import ujson
import numpy as np
import json
import jieba
import jieba.posseg as pseg
filename = r"E:\MedicalQA\MedicalQA\train\jieba_index_dict.ujson"
def read_from_ujson(fileName):
		print('load .. ' + fileName)
		fp = open(fileName,"rb")
		data =  ujson.load(fp)
		fp.close()
		print("load data done")
		return data

data = read_from_ujson(filename)
with open("dataaa.txt",'w',encoding = 'utf8') as f:
    f.write(json.dumps(data,ensure_ascii = False))
