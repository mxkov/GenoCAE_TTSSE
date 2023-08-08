import numpy as np
import tensorflow as tf
import utils.normalization as normalization


class data_generator_distrib:
	"""docstring"""

	def __init__(self, filebase,
	             normalization_mode="genotypewise01",
	             normalization_options={"flip": False,
	                                    "missing_val": 0.0},
	             impute_missing=True):
		self.filebase = filebase
		self.missing_val = normalization_options["missing_val"]
		self.normalization_mode = normalization_mode
		self.normalization_options = normalization_options
		self.impute_missing = impute_missing
		# maybe take & set more attrs here

		self._get_dims()
		self._define_samples()
		# no loading here

	def _define_samples(self):
		# TODO
		return

	def _get_dims(self):
		"""Count markers and samples"""
		# TODO
		return

	def _normalize(self):
		# TODO, as mapping
		return

	def _sparsify(self, mask, keep_fraction):
		# TODO, as mapping
		return

	def define_validation_set(self, validation_split=0.2):
		# TODO
		return

	# The key part (supposedly)
	def generator(self, pref_chunk_size, training, shuffle=True):
		# TODO
		# handle data loading
		# https://www.tensorflow.org/guide/data
		# https://stackoverflow.com/q/68164440
		return

	# Another key part
	def create_dataset(self, pref_chunk_size, shuffle=True):
		# TODO
		# feed self.generator to tf.data.Dataset.from_generator()
		# possibly with TFRecord: https://stackoverflow.com/q/59458298
		# see also https://www.tensorflow.org/guide/data_performance
		return

	# distribute dataset: https://stackoverflow.com/q/59185729