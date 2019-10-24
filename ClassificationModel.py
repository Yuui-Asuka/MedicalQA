import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
import numpy as np
import os
import time,datetime
import pandas as pd
import math
import gensim
from Config import Config

class ClassificationModel(object):
	def __init__(self,jiba_word_num,keyword_num):
		
		self.input_q = tf.placeholder(tf.int32, [None, None], name="input_q")#None
		self.input_a = tf.placeholder(tf.int32, [None, None], name="input_a")#None
		self.input_y = tf.placeholder(tf.float32, [None, Config.MAX_LABELS], name="input_y")

		self.seq_len_q = tf.placeholder(tf.int32, [None], name="seq_len_q")
		self.seq_len_a = tf.placeholder(tf.int32, [None], name="seq_len_a")
		self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
		self.batch_seq_len_q = tf.placeholder(tf.int32, name="batch_seq_len_q")
		self.batch_seq_len_a = tf.placeholder(tf.int32, name="batch_seq_len_a")
		embeddings_weights_q = tf.get_variable('embeddings_weights_q', shape=[jiba_word_num, Config.EMBEDDING_DIM], initializer=tf.uniform_unit_scaling_initializer())
		rnn_input_q = tf.nn.embedding_lookup(embeddings_weights_q, self.input_q)
		
		embeddings_weights_a = tf.get_variable('embeddings_weights_a', shape=[keyword_num, Config.EMBEDDING_DIM], initializer=tf.uniform_unit_scaling_initializer())
		rnn_input_a = tf.nn.embedding_lookup(embeddings_weights_a, self.input_a)
		
		#Bilstm layer(-s)
		dynamic_rnn_outputs_q = rnn_input_q
		for i in range(Config.NUM_LAYERS):
			with tf.variable_scope("bidirectional_rnn_q_%d" % i):
				(output, state) = tf.nn.bidirectional_dynamic_rnn(LSTMCell(Config.HIDDEN_SIZE), LSTMCell(Config.HIDDEN_SIZE),inputs=dynamic_rnn_outputs_q, sequence_length=self.seq_len_q, dtype=tf.float32)
				dynamic_rnn_outputs_q = tf.concat(output, 2)
				if i < Config.NUM_LAYERS-1:
					print("add dropout between lstm")
					dynamic_rnn_outputs_q = tf.nn.dropout(dynamic_rnn_outputs_q, self.dropout_keep_prob)

		# attention layer
		attention_output_q = self.dynamic_attention(dynamic_rnn_outputs_q, Config.ATTENTION_SIZE, self.batch_seq_len_q)
		
		#Bilstm layer(-s)
		dynamic_rnn_outputs_a = rnn_input_a
		for i in range(Config.NUM_LAYERS):
			with tf.variable_scope("bidirectional_rnn_a_%d" % i):
				(output, state) = tf.nn.bidirectional_dynamic_rnn(LSTMCell(Config.HIDDEN_SIZE), LSTMCell(Config.HIDDEN_SIZE),inputs=dynamic_rnn_outputs_a, sequence_length=self.seq_len_a, dtype=tf.float32)
				dynamic_rnn_outputs_a = tf.concat(output, 2)
				if i < Config.NUM_LAYERS-1:
					print("add dropout between lstm")
					dynamic_rnn_outputs_a = tf.nn.dropout(dynamic_rnn_outputs_a, self.dropout_keep_prob)

		# attention layer
		attention_output_a = self.dynamic_attention(dynamic_rnn_outputs_a, Config.ATTENTION_SIZE, self.batch_seq_len_a)
		attention_output = tf.concat([attention_output_q,attention_output_a],1)
		# Dropout
		drop_outputs = tf.nn.dropout(attention_output, self.dropout_keep_prob)
		
		print("fc layer intput dim:"+str(drop_outputs.get_shape()[1].value))

		drop_outputs_full = drop_outputs
		# fully connected layer
		with tf.name_scope("output_layer"):
			W = tf.get_variable(
				"output_weight",
				shape=[drop_outputs_full.get_shape()[1].value, Config.MAX_LABELS],
				initializer=tf.contrib.layers.xavier_initializer())
			b = tf.Variable(tf.constant(0.1, shape=[Config.MAX_LABELS]), name="output_b")
			logits = tf.nn.xw_plus_b(drop_outputs_full, W, b, name="y_matmul")
			self.pre_y = tf.nn.softmax(logits,name="pre_y")
			#self.pre_y = tf.argmax(logits_temp, 1)
		# define loss
		with tf.name_scope("loss"):
			self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.input_y))

		# define optimizer
		with tf.name_scope("optimizer"):
			self.optimizer = tf.train.AdamOptimizer(Config.learning_rate).minimize(self.loss)

		# define saver，only save last 5 models
		#saver = tf.train.Saver(tf.global_variables())
		self.saver = tf.train.Saver(tf.global_variables())
		self.predict_top = tf.nn.top_k(self.pre_y, k=5)
		self.predict_top1 = tf.nn.top_k(self.pre_y, k=1)
		self.label = tf.nn.top_k(self.input_y, k=1) 

	def dynamic_attention(self, inputs, attention_size,sequence_length):
	    if isinstance(inputs, tuple):
	        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
	        inputs = tf.concat(inputs, 2)

	    inputs_shape = inputs.shape #[batch_size,seq_len,hidden_size]
	    #sequence_length = inputs_shape[1].value  # the length of sequences processed in the antecedent RNN layer
	    hidden_size = inputs_shape[2].value  # hidden size of the RNN layer

	    # Attention mechanism
	    W = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
	    b = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
	    u = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
	    #[batch_size*seq_len,hidden_size] matmul [hidden_size,attention_size] = [batch_size*seq_len,attention_size]
	    #+ [1,attention_size]
	    #[batch_size*seq_len,attention_size] 
	    v = tf.tanh(tf.matmul(tf.reshape(inputs, [-1, hidden_size]), W) + tf.reshape(b, [1, -1]))
	    # [batch_size*seq_len,attention_size] matmul [attention_size,1] = [batch_size*seq_len,1]
	    vu = tf.matmul(v, tf.reshape(u, [-1, 1]))#[batch_size*seq_len,1]
	    #[batch_size,seq_len]
	    exps = tf.reshape(tf.exp(vu), [-1, sequence_length])#[batch_size,seq_len]
	    alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])#[batch_size,seq_len]
	    # Output of Bi-RNN is reduced with attention vector
	    #[batch_size,seq_len,hidden_size]*[batch_size,seq_len,1] = [batch_size,seq_len,hidden_size]
	    output = tf.reduce_sum(inputs * tf.reshape(alphas, [-1, sequence_length, 1]), 1)#[batch_size,hidden_size]
	    return output

	 
