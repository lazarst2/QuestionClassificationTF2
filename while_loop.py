import tensorflow as tf
import numpy as np
from bert import bert_models_library as bert_lib
from bert import metrics, modeling, bert_models

# =============================================================================
# def get_loss_function():
#     return tf.keras.losses.BinaryCrossentropy(from_logits=True)
# 
# loss_function = get_loss_function()
# 
# 
# 
# print(loss_function(np.array([[0, 1, 0]],dtype=np.float16),np.array( [[0.1,0.2,0.6]],dtype=np.float16)))
# =============================================================================
# bert_config = modeling.BertConfig.from_json_file("./bert/bert_default/bert_config.json")
# model_classifier, bert_model = bert_lib.BertMultiLabelClassifier.get(bert_config, tf.float32, 1917, 128)
# l = tf.train.list_variables("gs://cloud-tpu-checkpoints/bert/tf_20/uncased_L-12_H-768_A-12/bert_model.ckpt")
# #print(l)

# cm,bm = bert_models.classifier_model(bert_config, tf.float32, 1917, 128)
# res = [var.name for var in bert_model.trainable_variables]
# print(res)
# res = [var.name for var in bm.trainable_variables]
# print(res)
m = metrics.F1_Score()
m.update_state([[1,0,1],[1,1,0]],[[0,0,1],[1,1,0]])
print(m.result().numpy())
m.reset_states()