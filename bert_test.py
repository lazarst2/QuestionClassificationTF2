import tensorflow as tf
from bert import modeling
from bert import bert_models_library as bert_lib
import numpy as np


bert_config = modeling.BertConfig.from_json_file("./bert/bert_default/bert_config.json")
print(bert_config.__dict__)

encoder_config = bert_lib.EncoderConfig(max_number_documents=10,hidden_size=32)
print(encoder_config.__dict__)


brnn = bert_lib.BertGRUBidirectionalEncoder.get(
					bert_config=bert_config, 
					encoder_config=encoder_config,
					max_seq_len=3,
					float_type=tf.float16)


# ids = np.array([[1,2,3]])
# mask = np.array([[1,1,1]])
# in_ids = np.array([[0,0,0]])

# out = bert_layer(ids,mask,in_ids)
# print(out)

inputs = np.array([[[[1,2,3],[1,1,1],[0,0,0]]]], dtype=np.float16)
print(inputs.shape)
out = brnn.predict(inputs)
print(out)