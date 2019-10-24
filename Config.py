class Config:
	"""docstring for Config"""
	
	batch_size = 64

	MAX_LENGTH = 128
	MAX_LABELS = 23	 # 

	NUM_EPOCHS = 30 # 训练轮数

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

	path_cache_ques_all = "./cache/df_cutword.pkl"
	path_cache_answer_all = "./cache/qidAndAnswerMap.pkl"

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
