import tensorflow as tf


class F1_Score(tf.keras.metrics.Metric):
	
	def __init__(self, name="f1_score", **kwargs):
		super(F1_Score, self).__init__(name=name, **kwargs)
		self.recall = tf.keras.metrics.Recall(**kwargs)
		self.precision = tf.keras.metrics.Precision(**kwargs)
	

	def reset_states(self):
		self.recall.reset_states()
		self.precision.reset_states()
	

	def update_state(self, y_true, y_pred, sample_weight=None):
		self.recall.update_state(y_true,y_pred)
		self.precision.update_state(y_true,y_pred)
	

	def result(self):
		recall = self.recall.result()
		precision = self.precision.result()

		return 2*(precision*recall)/(precision+recall)