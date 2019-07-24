import tensorflow as tf
from bert import modeling
from bert import bert_models_library as bert_lib
import numpy as np


# class config(object):
#     def __init__(self,dense_units,recurent_units):
#         self.dense_units = dense_units
#         self.recurent_units = recurent_units

class MinimalRNNCell(tf.keras.layers.AbstractRNNCell):

    def __init__(self, recurrent_units, bert_config, **kwargs):
        self.bert_config = bert_config
        self.recurrent_units = recurrent_units
        super(MinimalRNNCell, self).__init__(**kwargs)

    @property
    def state_size(self):
      return self.recurrent_units
  
    def build(self, input_shape):
        self.bert_layer = bert_lib.BertLayer(config=self.bert_config, float_type=tf.float16)
        self.gru = tf.keras.layers.GRUCell(self.recurrent_units, dtype=tf.float16)
        super(MinimalRNNCell, self).build(input_shape)
        print(self.gru.dtype)
        
    def call(self, inputs, states):
        inputs = tf.cast(inputs,dtype=tf.int32)
        d_out = self.bert_layer(inputs[:,0], inputs[:,1], inputs[:,2])
        output, state = self.gru(d_out,states)
        
        return output, state
    def get_config(self):
        return {
            "bert_config":self.bert_config,
            "recurrent_units": self.recurrent_units
        }
    
# cell = MinimalRNNCell(config(16,8))
# x = tf.keras.Input((None, 5))
# inp = tf.ones(shape=[1,2,8])
# layer_backward = tf.keras.layers.RNN(cell,return_sequences=True, go_backwards=True)
# layer_forward = tf.keras.layers.RNN(cell,return_sequences=True, go_backwards=False)
# gru = tf.keras.layers.GRU(32,return_sequences=True)
# brnn = tf.keras.layers.Bidirectional(layer_forward, input_shape=(8, 8))
# y1 = brnn(inp)
# y2 = layer_forward(inp)

# print(y1)
# print(y2)

bert_config = modeling.BertConfig.from_json_file("./bert/bert_default/bert_config.json")
print(bert_config.__dict__)

cell = MinimalRNNCell(recurrent_units=16, bert_config=bert_config)
print(cell.state_size)
layer = tf.keras.layers.RNN(cell)
brnn = tf.keras.layers.Bidirectional(layer)


# ids = np.array([[1,2,3]])
# mask = np.array([[1,1,1]])
# in_ids = np.array([[0,0,0]])

# out = bert_layer(ids,mask,in_ids)
# print(out)

inputs = np.array([[[[1,2,3],[1,1,1],[0,0,0]]]], dtype=np.float16)
print(inputs.shape)
out = brnn(inputs)
print(out) 