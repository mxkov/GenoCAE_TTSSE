import glob
import numpy as np
import pyarrow.parquet as pq
import tensorflow as tf
import utils.normalization as normalization


class data_generator_distrib:
	"""docstring"""
	# TODO: go over all attrs used, make sure they exist & are set

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
	def generator(self, pref_chunk_size, training=True, shuffle=True):
		# handle data loading
		# https://www.tensorflow.org/guide/data
		# https://stackoverflow.com/q/68164440

		if training:
			n_samples = self.n_train_samples
			cur_sample_idx = self.sample_idx_train[np.arange(0, n_samples)]
		else:
			n_samples = self.n_valid_samples
			cur_sample_idx = self.sample_idx_valid[np.arange(0, n_samples)]
		if shuffle:
			cur_sample_idx = tf.random.shuffle(cur_sample_idx)
		cur_sample_idx = tf.cast(cur_sample_idx, tf.int32)

		# TODO: make sure to properly support multiple files, everywhere
		pq_paths = sorted(glob.glob(self.filebase+"*.parquet"))
		int32_t_MAX = 2**31-1
		pqds = pq.ParquetDataset(path_or_paths = pq_paths,
		                         thrift_string_size_limit = int32_t_MAX,
		                         thrift_container_size_limit = int32_t_MAX,
		                         use_legacy_dataset = False)
		# OBS! might not preserve column order. rely on schema instead.
		sch = pqds.schema

		chunk_size = pref_chunk_size - pref_chunk_size % self.batch_size
		num_chunks = np.ceil(n_samples / chunk_size)
		chunks_read = 0

		while chunks_read < num_chunks:

			start = chunk_size * chunks_read
			end   = chunk_size *(chunks_read+1)
			chunk_idx = cur_sample_idx[start:end]
			batches_per_chunk = np.ceil(len(chunk_idx)/self.batch_size)

			# TODO: store ind_pop list as an attr?
			inds_to_read = list(self.ind_pop_list[chunk_idx,0])
			data = pqds.read(columns = inds_to_read,
			                 use_threads = True,  # TODO: try without
			                 use_pandas_metadata = False)
			# TODO: convert this pa.Table to Numpy array
			# TODO: batching

		# TODO: the rest KEKW
		return

	# Another key part
	def create_dataset(self, pref_chunk_size, shuffle=True):
		# TODO
		# feed self.generator to tf.data.Dataset.from_generator()
		# possibly with TFRecord: https://stackoverflow.com/q/59458298
		# see also https://www.tensorflow.org/guide/data_performance
		return

	# distribute dataset: https://stackoverflow.com/q/59185729