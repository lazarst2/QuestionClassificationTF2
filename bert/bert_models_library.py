from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import tensorflow as tf
import six
import json

from bert import modeling



class EncoderConfig(object):
	def __init__(
			self,
			max_number_documents,
			hidden_size):

		self.max_number_documents = max_number_documents
		self.hidden_size = hidden_size
	@classmethod
	def from_dict(cls, json_object):
		config = cls(max_number_documents=None,hidden_size=None)
		for (key, value) in six.iteritems(json_object):
			config.__dict__[key] = value
		return config
	@classmethod
	def from_json_file(cls, json_file):
		"""Constructs a `BertConfig` from a json file of parameters."""
		with tf.io.gfile.GFile(json_file, "r") as reader:
			text = reader.read()
		return cls.from_dict(json.loads(text))


class BertLayer(tf.keras.layers.Layer):
	def __init__(self, config, float_type=tf.float32, **kwargs):
		super(BertLayer, self).__init__(**kwargs)
		self.config = (
			modeling.BertConfig.from_dict(config)
		if isinstance(config, dict) else copy.deepcopy(config))
		self.float_type = float_type

	
	def build(self,unused_input_shapes):
		self.bert_model_layer = modeling.BertModel(self.config,self.float_type)

		super(BertLayer, self).build(unused_input_shapes)
	

	def __call__(
			self,
			input_word_ids,
			input_mask=None,
			input_type_ids=None,
			**kwargs):

		inputs = modeling.pack_inputs([input_word_ids, input_mask, input_type_ids])
		return super(BertLayer, self).__call__(inputs, **kwargs)
	
	def call(self, inputs):
		unpacked_inputs = modeling.unpack_inputs(inputs)
		input_word_ids = unpacked_inputs[0]
		input_mask = unpacked_inputs[1]
		input_type_ids = unpacked_inputs[2]

		print(input_word_ids)

		pooled_output, sequence_output = self.bert_model_layer(input_word_ids, input_mask,input_type_ids)

		#Shape (batch_size, num_hidden_units)
		return pooled_output, sequence_output


class BertGRUEncoderCell(tf.keras.layers.AbstractRNNCell):
	def __init__(self, encoder_config, bert_config, float_type, **kwargs):
		super(BertGRUEncoderCell, self).__init__(**kwargs)
		self.bert_config = bert_config
		self.encoder_config = encoder_config
		self.float_type = float_type

	@property
	def state_size(self):
		return self.encoder_config.hidden_size
	


	def build(self, input_shape):
		self.bert_layer = BertLayer(config=self.bert_config, float_type=self.float_type)
		self.gru = tf.keras.layers.GRUCell(self.encoder_config.hidden_size, dtype=self.float_type)
		super(BertGRUEncoderCell, self).build(input_shape)


	def call(self, inputs, states):
		inputs = tf.cast(inputs, dtype=tf.int32)
		bert_output	= self.bert_layer(inputs[:,0], inputs[:,1], inputs[:,2])

		output, state = self.gru(bert_output, states)

		return output, state


	def get_config(self):
		return {
			"bert_config":self.bert_config,
			"encoder_config": self.encoder_config,
			"float_type":self.float_type}


class BertGRUBidirectionalEncoder(tf.keras.Model):

	def __init__(self, bert_config, encoder_config, float_type=tf.float32, **kwargs):
		super(BertGRUBidirectionalEncoder, self).__init__(**kwargs)

		self.bert_config = (
			modeling.BertConfig.from_dict(bert_config)
			if isinstance(bert_config, dict) else copy.deepcopy(bert_config))
		self.encoder_config = (
			EncoderConfig.from_dict(encoder_config)
			if isinstance(encoder_config, dict) else copy.deepcopy(encoder_config))

		self.float_type = float_type

		self.encoder_cell = BertGRUEncoderCell(
								encoder_config=self.encoder_config,
								bert_config=self.bert_config,
								float_type=float_type)

		self.recurrent_layer = tf.keras.layers.RNN(self.encoder_cell,return_sequences=False)
		self.bidirectional_rnn = tf.keras.layers.Bidirectional(self.recurrent_layer)


	#Time major shape input tensors
	#Shapes (number_documents, batch_size, max_seq_len)
	# def __call__(
	# 		self,
	# 		sequence_input_word_ids,
	# 		sequence_input_mask=None,
	# 		sequence_input_type_ids=None,
	# 		**kwargs):

	# 	inputs = modeling.pack_inputs([sequence_input_word_ids, sequence_input_mask, sequence_input_type_ids])
	# 	return super(BertEncoder, self).__call__(inputs, **kwargs)



	def call(self, inputs):
		inputs = tf.cast(inputs,dtype=self.float_type)
		return self.bidirectional_rnn(inputs)

	@staticmethod
	def get(bert_config, encoder_config, max_seq_len, float_type=tf.float32, **kwargs):
		inputs = tf.keras.layers.Input(shape=[None, 3, max_seq_len], dtype=tf.int32)

		model = BertGRUBidirectionalEncoder(
					bert_config=bert_config, 
					encoder_config=encoder_config, 
					float_type=tf.float16)

		outputs = model(inputs)

		return tf.keras.Model(
			inputs= inputs,
			outputs= outputs)


class GRUDecoder(tf.keras.Model):
	def __init__(self,**kwargs):
		pass
	def build(self, unused_input_shapes):

		super(GRUDecoder, self).build(unused_input_shapes)
	def call(self, input):
		pass


class BertMultiLabelClassifier(tf.keras.Model):
	def __init__(
			self,
			bert_config,
			float_type,
			num_labels,
			final_layer_initializer=None,
			**kwargs):
		super(BertMultiLabelClassifier, self).__init__(**kwargs)
		self.bert_encoder = BertLayer(bert_config,float_type,name="")
		self.dropout = tf.keras.layers.Dropout(rate=bert_config.hidden_dropout_prob)
		self.float_type = float_type
		if final_layer_initializer is not None:
			initializer = final_layer_initializer
		else:
			initializer = tf.keras.initializers.TruncatedNormal(
				stddev= bert_config.initializer_range)

		self.dense = tf.keras.layers.Dense(
			units=num_labels,
			kernel_initializer=initializer,
			name="output",
			dtype=float_type)
	def __call__(
			self,
			input_word_ids,
			input_mask=None,
			input_type_ids=None,
			**kwargs):

		inputs = modeling.pack_inputs([input_word_ids, input_mask, input_type_ids])
		return super(BertMultiLabelClassifier, self).__call__(inputs, **kwargs)
	
	def call(self, inputs):
		unpacked_inputs = modeling.unpack_inputs(inputs)
		input_word_ids = unpacked_inputs[0]
		input_mask = unpacked_inputs[1]
		input_type_ids = unpacked_inputs[2]

		self.pooled_output, _ = self.bert_encoder(input_word_ids, input_mask,input_type_ids)

		logits = self.dense(self.dropout(self.pooled_output))

		outputs = tf.keras.activations.sigmoid(logits)

		return logits


	@staticmethod
	def get(
			bert_config,
			float_type,
			num_labels,
			max_seq_length,
			final_layer_initializer=None):

		input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name='input_word_ids')
		input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name='input_mask')
		input_type_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name='input_type_ids')

		classifier = BertMultiLabelClassifier(bert_config,float_type,num_labels,final_layer_initializer)

		outputs =  classifier(input_word_ids, input_mask,input_type_ids)

		pooled_output, sequence_output = classifier.bert_encoder(input_word_ids, input_mask,input_type_ids)


		model = tf.keras.Model(
			inputs={
				'input_word_ids': input_word_ids,
				'input_mask': input_mask,
				'input_type_ids': input_type_ids
			},
			outputs=outputs)
		bert_model = tf.keras.Model(
	      inputs=[input_word_ids, input_mask, input_type_ids],
	      outputs=[pooled_output, sequence_output])

		return model, bert_model
	@staticmethod
	def get_loss_function():
		return tf.keras.losses.BinaryCrossentropy(from_logits=False,reduction=tf.keras.losses.Reduction.NONE)

	@staticmethod
	def get_metrics():
		return tf.keras.losses.BinaryCrossentropy(from_logits=False)