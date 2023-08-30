import glob
import numpy as np
import os
import pathlib
import pyarrow.parquet as pq
import scipy
import tensorflow as tf
import utils.normalization as normalization


def _bytes_feature(value):
	if isinstance(value, type(tf.constant(0))):
		value = value.numpy()
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def make_example(genos_, indpop_):
	features_ = {
		'len':    _int64_feature(genos_.shape[0]),
		'snp':    _bytes_feature(tf.io.serialize_tensor(genos_)),
		'indpop': _bytes_feature(tf.io.serialize_tensor(indpop_))
	}
	return tf.train.Example(features=tf.train.Features(feature=features_))

def decode_example(x, geno_dtype=np.float32):
	features_ = {
		'len':    tf.io.FixedLenFeature([], tf.int64),
		'snp':    tf.io.FixedLenFeature([], tf.string),
		'indpop': tf.io.FixedLenFeature([], tf.string)
	}
	example = tf.io.parse_single_example(x, features_)
	genos_  = tf.io.parse_tensor(example['snp'],
	                             out_type=tf.as_dtype(geno_dtype))
	indpop_ = tf.io.parse_tensor(example['indpop'],
	                             out_type=tf.string)
	return genos_, indpop_
# TODO: ^ move all these down, and encapsulate


class data_generator_distrib:
	"""docstring""" # TODO: DOCS & comments

	def __init__(self, filebase,
	             missing_mask=True, impute_missing=True,
	             normalization_mode="genotypewise01",
	             normalization_options={"flip": False,
	                                    "missing_val": -1.0}):
		self.filebase = filebase
		self.missing_mask = missing_mask
		self.impute_missing = impute_missing
		self.missing_val = normalization_options["missing_val"]
		self.normalization_mode = normalization_mode
		self.normalization_options = normalization_options
		# maybe take & set more attrs here
		self.total_chunks = None

		self._get_ind_pop_list()
		self._get_n_markers()
		self._define_samples()

	def _get_ind_pop_list(self):
		self.ind_pop_list = np.empty(shape=(0,2), dtype=str)
		fam_files = sorted(glob.glob(self.filebase+"*.fam"))
		for file in fam_files:
			indpop = np.genfromtxt(file, usecols=(1,0), dtype=str)
			self.ind_pop_list=np.concatenate((self.ind_pop_list,indpop), axis=0)

	def _get_n_markers(self):
		self.n_markers = 0
		bim_files = sorted(glob.glob(self.filebase+"*.bim"))
		for file in bim_files:
			self.n_markers += len(np.genfromtxt(file, usecols=(1), dtype=str))

	def _define_samples(self):
		self.n_total_samples = len(self.ind_pop_list)
		self.n_train_samples = len(self.ind_pop_list)
		self.n_valid_samples = 0

		self.sample_idx_all   = np.arange(self.n_total_samples)
		self.sample_idx_train = np.arange(self.n_train_samples)
		self.sample_idx_valid = np.arange(self.n_valid_samples)

		self.ind_pop_list_train = np.copy(self.ind_pop_list)
		self.ind_pop_list_valid = np.empty(shape=(0,2), dtype=str)


	def _define_validation_set(self, random_state=None):
		# TODO: should be stratified by population in the general case
		self.n_valid_samples = np.floor(self.n_total_samples*self.valid_split)
		self.n_train_samples = self.n_total_samples - self.n_valid_samples

		np.random.seed(random_state)
		self.sample_idx_valid = np.random.choice(self.sample_idx_all,
		                                         size=self.n_valid_samples,
		                                         replace=False)
		self.sample_idx_valid = np.sort(self.sample_idx_valid)
		train_idx = np.in1d(self.sample_idx_train, self.sample_idx_valid,
		                    invert=True)
		self.sample_idx_train = np.copy(self.sample_idx_train[train_idx])

		self.ind_pop_list_train=np.copy(self.ind_pop_list[self.sample_idx_train,:])
		self.ind_pop_list_valid=np.copy(self.ind_pop_list[self.sample_idx_valid,:])
		# TODO: see where else to insert np.copy


	# The most important method to be called from outer scope
	def create_tf_dataset(self, batch_size, outprefix,
	                      valid_split=0.2, valid_random_state=None,
	                      geno_dtype=np.float32, sparsifies=None,
	                      num_workers=1,
	                      batch_shards=False, pref_chunk_size=None,
	                      shuffle_tfrecords=False, shuffle_dataset=True,
	                      overwrite_tfrecords=False):

		self.batch_size   = batch_size
		self.outprefix    = outprefix
		self.valid_split  = valid_split
		self.geno_dtype   = geno_dtype
		self.sparsifies   = sparsifies
		self.num_workers  = num_workers
		self.batch_shards = batch_shards
		# TODO: could set one dtype for tfrecords and another for loading them

		existing_tfr_files = sorted(glob.glob(self.outprefix+"*.tfrecords"))
		if len(existing_tfr_files) == 0 or overwrite_tfrecords:
			self.parquet_to_tfrecords(pref_chunk_size_ = pref_chunk_size,
			                          shuffle_ = shuffle_tfrecords)

		current_tfr_files = sorted(glob.glob(self.outprefix+"*.tfrecords"))
		ds = tf.data.Dataset.from_tensor_slices(filenames=current_tfr_files)

		worker_id = int(os.environ["SLURM_PROCID"])
		ds = ds.shard(self.num_workers, worker_id)
		ds = ds.interleave(tf.data.TFRecordDataset,
		                   num_parallel_calls=tf.data.AUTOTUNE,
		                   cycle_length=self.num_workers, block_length=1)
		ds = ds.map(lambda d: decode_example(d, geno_dtype=self.geno_dtype),
		            num_parallel_calls=tf.data.AUTOTUNE)
		if shuffle_dataset:
			ds = ds.shuffle(buffer_size=self.n_total_samples)

		self._define_validation_set(random_state=valid_random_state)
		ds_train, ds_valid = self.finalize_train_valid_sets(ds)
		self.dataset_train = ds_train
		self.dataset_valid = ds_valid


	def parquet_to_tfrecords(self, pref_chunk_size_=None, shuffle_=False):
		if pref_chunk_size_ is None:
			# TODO: auto select
			pass

		gen_outshapes = (
			tf.TensorSpec(shape=(None, self.n_markers),
			              dtype=tf.as_dtype(self.geno_dtype)),
			tf.TensorSpec(shape=(None, 2), dtype=tf.string),
			tf.TensorSpec(shape=(1, 1), dtype=tf.bool)
		)
		gen_args = (pref_chunk_size_, shuffle_)
		ds = tf.data.Dataset.from_generator(self.generator_from_parquet,
		                                    output_signature=gen_outshapes,
		                                    args=gen_args)

		# TODO: if training, prep norm scaler
		# (if norm methods other than genotype-wise are applicable)

		ds = ds.map(self._normalize, num_parallel_calls=tf.data.AUTOTUNE)

		# Apparently can't write TFRecords in parallel.
		# TODO: make sure this works with multiprocessing (check in calling scope)
		if "SLURM_PROCID" in os.environ and int(os.environ["SLURM_PROCID"])==0:
			self.write_TFRecords(ds)


	def generator_from_parquet(self, pref_chunk_size_, shuffle_):
		# handle data loading
		# https://www.tensorflow.org/guide/data
		# https://stackoverflow.com/q/68164440

		n_samples = self.n_total_samples
		cur_sample_idx = tf.cast(self.sample_idx_all, tf.int32)
		if shuffle_:
			cur_sample_idx = tf.random.shuffle(cur_sample_idx)

		if self.batch_shards:
			gen_batch_size = min(self.batch_size, pref_chunk_size_)
		else:
			gen_batch_size = pref_chunk_size_

		# TODO: make sure to properly support multiple files, everywhere
		pq_paths = sorted(glob.glob(self.filebase+"*.parquet"))
		int32_t_MAX = 2**31-1
		pqds = pq.ParquetDataset(path_or_paths = pq_paths,
		                         thrift_string_size_limit = int32_t_MAX,
		                         thrift_container_size_limit = int32_t_MAX,
		                         use_legacy_dataset = False)
		# OBS! might not preserve column order. rely on schema instead.
		pqds_schema = pqds.schema
		inds_sch = [entry.name for entry in pqds_schema]
		inds_fam = list(self.ind_pop_list[:,0])
		if inds_sch != inds_fam:
			raise ValueError("Parquet schema inconsistent with FAM files")

		chunk_size = pref_chunk_size_ - pref_chunk_size_ % gen_batch_size
		self.total_chunks = np.ceil(n_samples / chunk_size)

		chunks_read = 0
		while chunks_read < self.total_chunks:

			start = chunk_size * chunks_read
			end   = chunk_size *(chunks_read+1)
			chunk_idx = cur_sample_idx[start:end,:]
			# Last chunk does not necessarily contain chunk_size samples!

			inds_to_read = list(self.ind_pop_list[chunk_idx,0])
			chunk = pqds.read(columns = inds_to_read,
			                  use_threads = True,  # TODO: try without
			                  use_pandas_metadata = False)
			sch   = chunk.schema
			chunk = chunk.to_pandas(self_destruct=True).to_numpy(dtype=self.geno_dtype)
			# TODO: if you use float16, other scripts should support that
			chunk = chunk.T
			assert chunk.shape[1] == self.n_markers
			assert [entry.name for entry in sch] == inds_to_read

			chunks_read += 1

			batches_read = 0
			batches_in_chunk = np.ceil(chunk.shape[0]/gen_batch_size)
			last_batch_in_chunk = False
			while batches_read < batches_in_chunk:

				start_ = gen_batch_size * batches_read
				end_   = gen_batch_size *(batches_read+1)
				batch = chunk[start_:end_]
				if end_ >= chunk.shape[0]:
					last_batch_in_chunk = True

				batch_idx    = cur_sample_idx[start_:end_]
				batch_indpop = self.ind_pop_list[batch_idx,:]

				batches_read += 1

				yield batch, batch_indpop, last_batch_in_chunk


	def _normalize(self, x, indpop, last_batch):
		"""normalize and insert missing value"""

		missing = tf.where(x == 9)
		a = tf.ones(shape=tf.shape(missing)[0], dtype=x.dtype)
		a = tf.sparse.SparseTensor(indices=missing, values=a,
		                           dense_shape=x.shape)

		if self.impute_missing:
			most_common_genos = self._get_most_common_genotypes(x)
			# TODO: this is batch level.
			b = tf.gather(most_common_genos, indices=missing[:,1])-9
			b = tf.sparse.SparseTensor(indices=missing, values=b,
			                           dense_shape=x.shape)
			x = tf.sparse.add(x, b)

		if self.normalization_mode == "genotypewise01":
			if self.normalization_options["flip"]:
				x = -(x-2)/2
				if not self.impute_missing:
					x = tf.sparse.add(x, a*(3.5-self.missing_val))
					# TODO: do we flip the missing value here too?
			else:
				x = x/2
				if not self.impute_missing:
					x = tf.sparse.add(x, a*(self.missing_val-4.5))

		elif self.normalization_mode in ("standard", "smartPCAstyle"):
			raise NotImplementedError("Only genotypewise01 normalization "+
			                          "method supported for now")

		return x, indpop, last_batch


	def _get_most_common_genotypes(self, x_):
		return scipy.stats.mode(x_).mode[0]


	def _mask_and_sparsify(self, x, indpop):

		sparsify = False
		if self.sparsifies is not None and len(self.sparsifies)>0:
			sparsify_fraction = np.random.choice(self.sparsifies)
			if sparsify_fraction > 0.0:
				sparsify = True

		if not sparsify and not self.missing_mask:
			inputs = tf.expand_dims(x, axis=-1)
			return inputs, x, indpop

		inputs = tf.identity(x)
		mask   = tf.experimental.numpy.full(shape=x.shape, fill_value=1,
		                                    dtype=x.dtype)

		if sparsify:
			probs = tf.random.uniform(shape=x.shape, minval=0, maxval=1)
			where_sparse = tf.where(probs < sparsify_fraction)
			delta_s = tf.repeat(-1, tf.shape(where_sparse)[0])
			delta_s = tf.sparse.SparseTensor(indices=where_sparse,
			                                 values=delta_s,
			                                 dense_shape=mask.shape)
			mask = tf.sparse.add(mask, delta_s)
			inputs = tf.math.add(tf.math.multiply(inputs, mask),
			                     self.missing_val*(1-mask))

		if self.missing_mask:
			if not self.impute_missing:
				where_missing = tf.where(x == self.missing_val)
				delta_m = tf.repeat(0, tf.shape(where_missing)[0])
				delta_m = tf.sparse.SparseTensor(indices=where_missing,
				                                 values=delta_m,
				                                 dense_shape=mask.shape)
				mask = tf.sparse.sparse_dense_matmul(delta_m, mask)
			inputs = tf.stack([inputs, mask], axis=-1)
		else:
			inputs = tf.expand_dims(inputs, axis=-1)

		return inputs, x, indpop
		# TODO: mind that x are targets. mb make this clearer in the code.


	def write_TFRecords(self, dataset):
		outdir = pathlib.Path(self.outprefix).parent
		if not os.path.isdir(outdir):
			os.makedirs(outdir)

		# Recommended: >10 MB per shard, but <10 shards per worker.
		# However, if chunk size is close to memory size,
		# then shards cannot be larger than chunks.
		#
		# For now, going with precisely one chunk per shard.
		# Will have to see how this works out.

		batch_count = 0
		shard_count = 0
		total_shards = self.total_chunks
		zfill_len = len(str(total_shards-1))
		for batch, batch_indpop, last_batch_in_chunk in dataset:
			if batch_count == 0:
				genos  = batch
				indpop = batch_indpop
			else:
				genos  = tf.concat([genos,  batch], axis=0)
				indpop = tf.concat([indpop, batch_indpop], axis=0)
			batch_count += 1

			if not last_batch_in_chunk:
				continue

			shard_id = str(shard_count).zfill(zfill_len)
			shard_filename = f"{self.outprefix}_{shard_id}.tfrecords"
			writer = tf.io.TFRecordWriter(shard_filename)
			# TODO: compression maybe?

			for i in range(genos.shape[0]):
				cur_geno = tf.cast(genos[i,:], tf.as_dtype(self.geno_dtype))
				example = make_example(cur_geno, indpop[i,:])
				writer.write(example.SerializeToString())

			batch_count  = 0
			shard_count += 1
			writer.close()


	def finalize_train_valid_sets(self, dataset):

		def filter_by_indpop(ds_, ind_pop_list_):
			return ds_.filter(lambda x,y: y.numpy().astype("str") \
			                              in ind_pop_list_)

		def check_size(ds_, ref_size, label):
			size = sum([1 for x,y in ds_.as_numpy_iterator()])
			if size != ref_size:
				raise ValueError(f"Wrong {label} dataset size: "+
				                 f"got {size}, expected {ref_size}")

		def finalize_split(ds_):
			ds_ = ds_.prefetch(tf.data.AUTOTUNE)
			ds_ = ds_.batch(self.batch_size//self.num_workers)
			ds_ = ds_.map(self._mask_and_sparsify,
			              num_parallel_calls=tf.data.AUTOTUNE)
			return ds_

		if self.n_valid_samples == 0:
			ds_train_ = dataset
			ds_valid_ = None
		else:
			ds_train_ = filter_by_indpop(dataset, self.ind_pop_list_train)
			ds_valid_ = filter_by_indpop(dataset, self.ind_pop_list_valid)

		check_size(ds_train_, self.n_train_samples, "training")
		ds_train_ = finalize_split(ds_train_)
		if self.n_valid_samples > 0:
			check_size(ds_valid_, self.n_valid_samples, "validation")
			ds_valid_ = finalize_split(ds_valid_)

		return ds_train_, ds_valid_
