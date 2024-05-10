from dataclasses import dataclass, field
import glob
from math import floor
import numpy as np
import os
import pathlib
from psutil import virtual_memory
import pyarrow.parquet as pq
import re
import scipy
import tensorflow as tf
from datetime import datetime

# TODO: backwards compatibility with PLINK / EIGENSTRAT
# TODO: also, doesn't make sense to convert to tfrecords if you only want to project.
#       feed a `training` arg to create_tf_dataset and add an option to skip that.

class DebugReport:

	def __init__(self):
		self._report = ""

	def add(self, addition):
		try:
			self._report += str(addition)
		except:
			pass

	def write(self, filename, directory="ae_reports", mode="w"):
		if mode not in ("w", "a"):
			mode = "w"
		os.makedirs(directory, exist_ok=True)
		f = open(os.path.join(directory, filename), mode)
		f.write(self._report)
		f.close()


@dataclass
class DataGenerator:
	"""docstring""" # TODO: DOCS & comments
	filebase:              str
	global_batch_size:     int
	tfrecords_prefix:      str  = ""
	drop_inds_file:        str  = None
	geno_dtype:            type = np.float32
	missing_mask:          bool = True
	_debug:                bool = False
	# Validation options
	valid_split:           float= 0.2
	valid_random_state:    int  = None
	# Pre-processing options
	impute_missing:        bool = False
	normalization_mode:    str  = "genotypewise01"
	normalization_options: dict = field(default_factory=lambda: {
	                              "flip": False, "missing_val": -1.0})
	sparsifies:            list[float] = field(default_factory=list)
	# Parquet-to-TFRecords conversion options
	pref_chunk_size:       int  = None
	batch_shards:          bool = False
	overwrite_tfrecords:   bool = False
	# tf.data.Dataset options
	shuffle_dataset:       bool = True
	shuffle_dataset_seed:  int  = None

	def __post_init__(self):
		self.missing_val  = self.normalization_options["missing_val"]
		self.chunk_size   = None
		self.total_chunks = None
		self.filelist = []
		self.samples_per_file = dict()

		if self._debug:
			self.debug_report = DebugReport()

		self._get_ind_pop_list()
		self._get_n_markers()
		self._define_samples()

		self.define_validation_set(validation_split = self.valid_split,
		                           random_state = self.valid_random_state)


	def _get_ind_pop_list(self):
		if self.drop_inds_file is not None:
			dropdata  = np.genfromtxt(self.drop_inds_file, usecols=(0,1),
			                          dtype=str, delimiter=",")
			drop_cols = dropdata[:,0].astype(np.int32)
			drop_vals = dropdata[:,1]
		else:
			dropdata  = None

		def drop_inds(indpop_):
			for i in range(len(drop_cols)):
				keep = indpop_[:,drop_cols[i]]!=drop_vals[i]
				indpop_ = indpop_[keep]
			return indpop_

		self.ind_pop_list = np.empty(shape=(0,2), dtype=str)
		fam_files = sorted(glob.glob(self.filebase+"*.fam"))
		if len(fam_files) == 0:
			raise FileNotFoundError("No FAM files found that fit " +
			                       f"the pattern {self.filebase}*.fam")

		if self._debug:
			self.debug_report.add("\nFAM files:\n"+"\n".join(fam_files)+"\n")

		for fam_file in fam_files:
			pq_file = re.sub(".fam$", ".parquet", fam_file)
			if not os.path.isfile(pq_file):
				raise FileNotFoundError(f"FAM File {fam_file} " +
				                         "does not have a corresponding " +
				                        f"Parquet file {pq_file}")

			indpop = np.genfromtxt(fam_file, dtype=str)
			if dropdata:
				indpop = drop_inds(indpop)
			indpop = indpop[:,[1,0]]
			self.ind_pop_list=np.concatenate((self.ind_pop_list,indpop), axis=0)

			self.filelist.append(pq_file)
			self.samples_per_file[pq_file] = indpop.shape[0]

		if self.ind_pop_list.shape[0] == 0:
			raise IOError(f"No samples found in FAM files {self.filebase}*.fam")

		if self._debug:
			self.debug_report.add(f"\nPQ filelist: {self.filelist}\n")


	def _get_n_markers(self):
		self.n_markers = 0
		bim_files = sorted(glob.glob(self.filebase+"*.bim"))
		for file in bim_files:
			self.n_markers += len(np.genfromtxt(file, usecols=(1), dtype=str))
		if self.n_markers <= 0:
			raise IOError(f"No markers found in BIM files {self.filebase}*.bim")

	def _define_samples(self):
		self.n_total_samples = len(self.ind_pop_list)
		self.n_train_samples = len(self.ind_pop_list)
		self.n_valid_samples = 0

		self.sample_idx_all   = dict()
		self.sample_idx_train = dict()
		self.sample_idx_valid = dict()
		start = 0
		for f in self.filelist:
			end = start + self.samples_per_file[f]
			self.sample_idx_all[  f] = np.arange(start, end)
			self.sample_idx_train[f] = np.arange(start, end)
			self.sample_idx_valid[f] = np.arange(0)
			start = end
			assert len(self.sample_idx_all[f]) == self.samples_per_file[f]

		self.ind_pop_list_train = np.copy(self.ind_pop_list)
		self.ind_pop_list_valid = np.empty(shape=(0,2), dtype=str)


	# The most important method to be called from outer scope
	def create_tf_dataset(self, input_context):
		# input_context:
		# https://www.tensorflow.org/api_docs/python/tf/distribute/InputContext
		# https://www.tensorflow.org/tutorials/distribute/input#tfdistributestrategydistribute_datasets_from_function
		#
		num_workers = input_context.num_input_pipelines
		worker_id   = input_context.input_pipeline_id
		batch_size  = input_context.get_per_replica_batch_size(self.global_batch_size)
		#num_devices = input_context.num_replicas_in_sync
		# TODO: could set one dtype for tfrecords and another for loading them

		# reset train/valid split
		self._define_samples()

		existing_tfr_files=sorted(glob.glob(self.tfrecords_prefix+"*.tfrecords"))
		if len(existing_tfr_files) == 0 or self.overwrite_tfrecords:
			self.parquet_to_tfrecords()

		current_tfr_files=sorted(glob.glob(self.tfrecords_prefix+"*.tfrecords"))
		ds = tf.data.Dataset.from_tensor_slices(current_tfr_files)

		ds = ds.shard(num_workers, worker_id)
		ds = ds.interleave(tf.data.TFRecordDataset,
		                   num_parallel_calls=tf.data.AUTOTUNE,
		                   cycle_length=num_workers, block_length=1)
		ds = ds.map(lambda d: decode_example(d, geno_dtype=self.geno_dtype),
		            num_parallel_calls=tf.data.AUTOTUNE)
		if self.shuffle_dataset:
			ds = ds.shuffle(buffer_size = self.n_total_samples,
			                seed = self.shuffle_dataset_seed)

		self.define_validation_set(validation_split = self.valid_split,
		                           random_state = self.valid_random_state)
		ds_train, ds_valid = self.finalize_train_valid_sets(ds, batch_size)
		self.dataset_train = ds_train
		self.dataset_valid = ds_valid


	def create_dataset_from_pq(self, input_context, split="all"):

		if split not in ("all", "train", "valid"):
			raise ValueError(f"Invalid split argument ({split}): "+
			                 "must be 'all', 'train' or 'valid'")

		if self.pref_chunk_size is None:
			self.pref_chunk_size = auto_chunk_size(width=self.n_markers,
			                                       dtype=self.geno_dtype)

		def pq_generator_wrapper(files):
			gen_outshapes = (
				# genotypes
				tf.TensorSpec(shape=(None, self.n_markers),
				              dtype=tf.as_dtype(self.geno_dtype)),
				# ind/pop
				tf.TensorSpec(shape=(None, 2), dtype=tf.string),
				# genotypes shape
				tf.TensorSpec(shape=(2,), dtype=tf.int64),
				# last batch flag
				tf.TensorSpec(shape=(1,), dtype=tf.bool)
			)
			gen_args = (files, split, batch_size,
			            self.pref_chunk_size, self.shuffle_dataset)
			ds_ = tf.data.Dataset.from_generator(self.generator_from_parquet,
			                                     output_signature=gen_outshapes,
			                                     args=gen_args)
			return ds_

		num_workers = input_context.num_input_pipelines
		worker_id   = input_context.input_pipeline_id
		batch_size  = input_context.get_per_replica_batch_size(self.global_batch_size)
		num_devices = input_context.num_replicas_in_sync

		pq_paths = self.filelist
		if len(pq_paths) % num_workers != 0:
			raise RuntimeError(f"Can't distribute {len(pq_paths)} input files" +
			                   f" evenly between {num_workers} workers")

		ds = tf.data.Dataset.from_tensor_slices(pq_paths)
		ds = ds.shard(num_workers, worker_id)

		if self._debug:
			self.debug_report.add(f"\nPQ paths to be sharded: {pq_paths}\n")
			self.debug_report.add(f"num_workers: {num_workers}\n"+
			                      f"worker_id: {worker_id}\n"+
			                      f"Sharded PQ paths:\n"+
			                      "\n".join([str(elem) for elem in ds])+"\n")

		ds = ds.interleave(pq_generator_wrapper,
		                   num_parallel_calls=tf.data.AUTOTUNE,
		                   cycle_length=tf.data.AUTOTUNE,
		                   block_length=1)
		# cycle_length is how many files are worked concurrently ON THIS WORKER.
		# source: debug reports.
		ds = ds.prefetch(tf.data.AUTOTUNE)
		ds = ds.map(self._normalize, num_parallel_calls=tf.data.AUTOTUNE)
		ds = ds.map(self._mask_and_sparsify,
		            num_parallel_calls=tf.data.AUTOTUNE)

		if self._debug:
			self.debug_report.write("general.txt")

		return ds


	def parquet_to_tfrecords(self):

		pq_paths = sorted(glob.glob(self.filebase+"*.parquet"))

		if self.pref_chunk_size is None:
			self.pref_chunk_size = auto_chunk_size(width=self.n_markers,
			                                       dtype=self.geno_dtype)
		if self.batch_chunks:
			gen_batch_size_ = min(self.global_batch_size, self.pref_chunk_size)
		else:
			gen_batch_size_ = self.pref_chunk_size

		gen_outshapes = (
			# genotypes
			tf.TensorSpec(shape=(None, self.n_markers),
			              dtype=tf.as_dtype(self.geno_dtype)),
			# ind/pop
			tf.TensorSpec(shape=(None, 2), dtype=tf.string),
			# genotypes shape
			tf.TensorSpec(shape=(2,), dtype=tf.int64),
			# last batch flag
			tf.TensorSpec(shape=(1,), dtype=tf.bool)
		)
		gen_args = (pq_paths, "all", gen_batch_size_,
		            self.pref_chunk_size, False)
		# TODO: apparently this makes all workers read all parquet files -_-
		#       possible solution: shard list on input files, then interleave
		#       with generator_from_parquet but make it take a list of files
		ds = tf.data.Dataset.from_generator(self.generator_from_parquet,
		                                    output_signature=gen_outshapes,
		                                    args=gen_args)

		# This spot is where we would prepare normalization scaler,
		# if normalization methods other than genotype-wise were applicable.

		ds = ds.map(self._normalize, num_parallel_calls=tf.data.AUTOTUNE)

		# Apparently can't write TFRecords in parallel.
		# TODO: make sure this works with multiprocessing (check in calling scope)
		if "SLURM_PROCID" in os.environ and int(os.environ["SLURM_PROCID"])==0:
			self.write_TFRecords(ds)


	def generator_from_parquet(self, filepaths, split_, gen_batch_size,
	                           pref_chunk_size_, shuffle_):
		# handle data loading
		# https://www.tensorflow.org/guide/data
		# https://stackoverflow.com/q/68164440

		if type(filepaths) == list and type(filepaths[0]) == bytes:
			filepaths = [f.decode() for f in filepaths]
		elif type(filepaths) == bytes:
			filepaths = [filepaths.decode()]
		else:
			raise TypeError(f"Unsupported filepaths type: {type(filepaths)}")
			# TODO: type all function args, actually

		if type(split_) == bytes:
			split_ = split_.decode()
		if split_ == "all":
			cur_sample_idx_per_file = self.sample_idx_all
		elif split_ == "train":
			cur_sample_idx_per_file = self.sample_idx_train
		elif split_ == "valid":
			cur_sample_idx_per_file = self.sample_idx_valid
		else:
			raise ValueError(f"Invalid split_ argument ({split_}): "+
			                 "must be 'all', 'train' or 'valid'")
		cur_sample_idx = np.concatenate([cur_sample_idx_per_file[f]
		                                 for f in filepaths], axis=0)
		cur_ind_pop_list = np.copy(self.ind_pop_list[cur_sample_idx,:])

		if self._debug:
			try:
				jid = os.environ["SLURM_JOB_ID"]
				wid = os.environ["SLURMD_NODENAME"]
			except:
				jid = "localjob"
				wid = "localnode"
			fname = re.sub(".parquet$", "", pathlib.Path(filepaths[0]).name)
			pqreport_file = f"slurm-{jid}_{wid}_{split_}_{fname}.txt"
			pqreport = DebugReport()
			pqreport.add(f"\nOpening on worker {wid} at {datetime.now().time()}:"+
			             f" {filepaths}\n")
			pqreport.write(pqreport_file, mode="a")

		# TODO: make sure to properly support multiple files, everywhere
		int32_t_MAX = 2**31-1
		pqds = pq.ParquetDataset(path_or_paths = filepaths,
		                         thrift_string_size_limit = int32_t_MAX,
		                         thrift_container_size_limit = int32_t_MAX,
		                         use_legacy_dataset = False)
		# OBS! might not preserve column order. rely on schema instead.
		inds_sch = np.array([entry.name for entry in pqds.schema])
		inds_fam = cur_ind_pop_list[:,0]
		present_in_pq  = np.in1d(inds_fam, inds_sch, invert=False)
		present_in_fam = np.in1d(inds_sch, inds_fam, invert=False)
		# TODO: this check might need a rework.
		if (inds_sch[present_in_fam] != inds_fam[present_in_pq]).any():
			raise ValueError("Parquet schema inconsistent with FAM files")

		assert (cur_sample_idx[present_in_pq] == cur_sample_idx).all()
		cur_sample_idx = tf.cast(cur_sample_idx, tf.int32)
		if shuffle_:
			cur_sample_idx = tf.random.shuffle(cur_sample_idx)
		n_samples = len(cur_sample_idx)

		chunk_size = pref_chunk_size_ - pref_chunk_size_ % gen_batch_size
		self.chunk_size = chunk_size
		self.total_chunks = np.ceil(n_samples / chunk_size).astype(int)

		if self._debug:
			pqreport.add(f"N samples: {n_samples}\n"+
			             f"Final chunks size: {self.chunk_size}\n"+
			             f"Total chunks: {self.total_chunks}\n")
			pqreport.write(pqreport_file, mode="a")

		chunks_read = 0
		total_sample_count = 0
		total_batch_count = 0
		while chunks_read < self.total_chunks:

			start = chunk_size * chunks_read
			end   = chunk_size *(chunks_read+1)
			chunk_idx = cur_sample_idx[start:end]
			# Last chunk does not necessarily contain chunk_size samples!

			chunk_indpop = self.ind_pop_list[chunk_idx,:]
			# chunk_indpop = cur_ind_pop_list[chunk_idx,:] -??
			inds_to_read = list(chunk_indpop[:,0])
			chunk = pqds.read(columns = inds_to_read,
			                  use_threads = True,  # TODO: try without
			                  use_pandas_metadata = False)
			assert [entry.name for entry in chunk.schema] == inds_to_read
			chunk = chunk.to_pandas(self_destruct=True).to_numpy(dtype=self.geno_dtype)
			# TODO: if you use float16, other scripts should support that
			chunk = chunk.T
			if chunk.shape[1] != self.n_markers:
				raise ValueError("Parquet table size inconsistent with BIM "+
				                f"files: {self.n_markers} markers expected "+
				                f"but {chunk.shape[1]} markers read")
			chunks_read += 1
			# TODO: this sanity check needs to account for dropped inds

			batches_read = 0
			last_batch_in_chunk = False
			while not last_batch_in_chunk:

				start_ = gen_batch_size * batches_read
				end_   = gen_batch_size *(batches_read+1)

				batch_genos  = chunk[start_:end_,:]
				batch_indpop = chunk_indpop[start_:end_,:]
				if end_ >= chunk.shape[0]:
					last_batch_in_chunk = True

				batches_read += 1
				total_batch_count += 1
				total_sample_count += batch_genos.shape[0]

				yield batch_genos, batch_indpop, \
				      batch_genos.shape, np.array([last_batch_in_chunk])

		if self._debug:
			pqreport.add(f"Finished at {datetime.now().time()}.\n"+
			             f"Total batches read: {total_batch_count}\n"+
			             f"Total samples read: {total_sample_count}\n")
			pqreport.write(pqreport_file, mode="a")


	def _normalize(self, genos, indpop, genos_shape, *args):
		"""normalize and insert missing value"""

		where_missing = tf.where(genos==9)
		a = tf.ones(shape=tf.shape(where_missing)[0], dtype=genos.dtype)
		a = tf.sparse.SparseTensor(indices=where_missing, values=a,
		                           dense_shape=genos_shape)

		if self.impute_missing:
			most_common_genos = get_most_common_genotypes(genos)
			b = tf.gather(most_common_genos, indices=where_missing[:,1])-9
			b = tf.sparse.SparseTensor(indices=where_missing, values=b,
			                           dense_shape=genos_shape)
			genos = tf.sparse.add(genos, b)

		if self.normalization_mode == "genotypewise01":
			if self.normalization_options["flip"]:
				genos = -(genos-2)/2
				# Missing genotypes will turn from 9 into -3.5,
				# need to replace those with missing_val.
				if not self.impute_missing:
					genos = tf.sparse.add(genos, a*(3.5+self.missing_val))
					# Thus, missing_val is NOT being flipped,
					# which should be taken into account when setting it.
			else:
				genos = genos/2
				# Missing genotypes will turn from 9 into 4.5,
				# need to replace those with missing_val.
				if not self.impute_missing:
					genos = tf.sparse.add(genos, a*(self.missing_val-4.5))

		elif self.normalization_mode in ("standard", "smartPCAstyle"):
			raise NotImplementedError("Only genotypewise01 normalization "+
			                          "method supported for now")
		else:
			raise ValueError("Unknown normalization mode: "+
			                f"{self.normalization_mode}")

		return genos, indpop, genos_shape, *args


	def write_TFRecords(self, dataset):
		outdir = pathlib.Path(self.tfrecords_prefix).parent
		if not os.path.isdir(outdir):
			os.makedirs(outdir)

		# Recommended: >10 MB per shard, but <10 shards per worker.
		# However, if chunk size is close to memory size,
		# then shards cannot be larger than chunks.
		#
		# For now, going with precisely one chunk per shard.
		# Will have to see how this works out.
		# Ideally, should have an option to further split a chunk into shards.
		# For now, will just read smaller chunks if more shards are needed.

		# TODO: mb check/ensure that num_shards > num_workers.
		#       could also rely on setting num_shards rather than chunk_size.
		#       or, account for both when auto-chunking.

		batch_count = 0
		shard_count = 0
		total_shards = self.total_chunks
		zfill_len = len(str(total_shards-1))
		for batch_genos, batch_indpop, _, last_batch_in_chunk in dataset:
			if batch_count == 0:
				genos  = batch_genos
				indpop = batch_indpop
			else:
				genos  = tf.concat([genos,  batch_genos ], axis=0)
				indpop = tf.concat([indpop, batch_indpop], axis=0)
			batch_count += 1

			if not last_batch_in_chunk:
				continue

			shard_id = str(shard_count).zfill(zfill_len)
			shard_filename = f"{self.tfrecords_prefix}_{shard_id}.tfrecords"
			writer = tf.io.TFRecordWriter(shard_filename)
			# TODO: compression maybe?

			for i in range(genos.shape[0]):
				cur_geno = tf.cast(genos[i,:], tf.as_dtype(self.geno_dtype))
				example = make_example(cur_geno, indpop[i,:])
				writer.write(example.SerializeToString())

			batch_count  = 0
			shard_count += 1
			writer.close()


	def define_validation_set(self, validation_split, random_state=None):
		# TODO: should be stratified by population in the general case
		self.n_valid_samples = 0
		self.n_train_samples = 0

		np.random.seed(random_state)
		for f in self.filelist:
			n_valid = floor(self.samples_per_file[f] * validation_split)
			n_train = self.samples_per_file[f] - n_valid
			self.sample_idx_valid[f] = np.random.choice(self.sample_idx_all[f],
			                                            size=n_valid,
			                                            replace=False)
			self.sample_idx_valid[f] = np.sort(self.sample_idx_valid[f])
			train_idx = np.in1d(self.sample_idx_all[f],
			                    self.sample_idx_valid[f],
			                    invert=True)
			self.sample_idx_train[f] = np.copy(self.sample_idx_all[f][train_idx])
			self.sample_idx_train[f] = np.sort(self.sample_idx_train[f])

			self.n_valid_samples += n_valid
			self.n_train_samples += n_train
			assert len(self.sample_idx_train[f]) == n_train

		assert self.n_train_samples == self.n_total_samples-self.n_valid_samples

		sample_idx_train_all = np.concatenate([self.sample_idx_train[f]
		                                       for f in self.filelist], axis=0)
		sample_idx_valid_all = np.concatenate([self.sample_idx_valid[f]
		                                       for f in self.filelist], axis=0)
		self.ind_pop_list_train=np.copy(self.ind_pop_list[sample_idx_train_all,:])
		self.ind_pop_list_valid=np.copy(self.ind_pop_list[sample_idx_valid_all,:])


	def finalize_train_valid_sets(self, dataset, batch_size_):

		def filter_by_indpop(ds_, ind_pop_list_):
			return ds_.filter(lambda g,indpop,g_shape: \
			                    indpop.numpy().astype("str") in ind_pop_list_)

		def check_size(ds_, ref_size, label):
			size = sum([1 for g,indpop,g_shape in ds_.as_numpy_iterator()])
			if size != ref_size:
				raise ValueError(f"Wrong {label} dataset size: "+
				                 f"got {size} samples, "+
				                 f"expected {ref_size} samples")

		def finalize_split(ds_):
			ds_ = ds_.prefetch(tf.data.AUTOTUNE)
			ds_ = ds_.batch(batch_size_)
			ds_ = ds_.map(self._mask_and_sparsify,
			              num_parallel_calls=tf.data.AUTOTUNE)
			return ds_

		if self.n_valid_samples > 0:
			ds_train_ = filter_by_indpop(dataset, self.ind_pop_list_train)
			ds_valid_ = filter_by_indpop(dataset, self.ind_pop_list_valid)
		else:
			ds_train_ = dataset
			ds_valid_ = None

		check_size(ds_train_, self.n_train_samples, "training")
		ds_train_ = finalize_split(ds_train_)
		if self.n_valid_samples > 0:
			check_size(ds_valid_, self.n_valid_samples, "validation")
			ds_valid_ = finalize_split(ds_valid_)

		return ds_train_, ds_valid_


	def _mask_and_sparsify(self, genos, indpop, genos_shape, *args):

		sparsify = False
		if len(self.sparsifies) > 0:
			sparsify_fraction = np.random.choice(self.sparsifies)
			if sparsify_fraction > 0.0:
				sparsify = True

		if not sparsify and not self.missing_mask:
			inputs = tf.expand_dims(genos, axis=-1)
			return inputs, genos, indpop

		inputs = tf.identity(genos)
		mask   = tf.cast(tf.fill(genos_shape, 1), tf.as_dtype(self.geno_dtype))
		if self.impute_missing:
			orig_mask = tf.identity(mask)
		else:
			where_nonmissing = tf.where(genos != self.missing_val)
			delta_m = tf.repeat(1, tf.shape(where_nonmissing)[0])
			delta_m = tf.sparse.SparseTensor(indices=where_nonmissing,
			                                 values=delta_m,
			                                 dense_shape=genos_shape)
			delta_m = tf.sparse.to_dense(tf.cast(delta_m, mask.dtype))
			orig_mask = delta_m

		if sparsify:
			probs = tf.random.uniform(shape=genos_shape, minval=0, maxval=1)
			where_sparse = tf.where(probs < sparsify_fraction)
			delta_s = tf.repeat(-1, tf.shape(where_sparse)[0])
			delta_s = tf.sparse.SparseTensor(indices=where_sparse,
			                                 values=delta_s,
			                                 dense_shape=genos_shape)
			delta_s = tf.cast(delta_s, mask.dtype)
			mask = tf.sparse.add(mask, delta_s)
			inputs = tf.math.add(tf.math.multiply(inputs, mask),
			                     self.missing_val*(1-mask))

		if self.missing_mask:
			if not self.impute_missing:
				mask = tf.math.multiply(mask, delta_m)
			inputs = tf.stack([inputs, mask], axis=-1)
		else:
			inputs = tf.expand_dims(inputs, axis=-1)

		orig_mask = tf.cast(orig_mask, tf.bool)

		# genos will serve as targets
		return inputs, genos, orig_mask, indpop, *args


def get_most_common_genotypes(genos_):
	return scipy.stats.mode(genos_).mode[0]


def make_example(genos_, indpop_):

	def int64_feature(value):
		return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

	def bytes_feature(value):
		if isinstance(value, type(tf.constant(0))):
			value = value.numpy()
		return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

	features_ = {
		'len':    int64_feature(genos_.shape[0]),
		'snp':    bytes_feature(tf.io.serialize_tensor(genos_)),
		'indpop': bytes_feature(tf.io.serialize_tensor(indpop_))
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
	return genos_, indpop_, tf.constant(genos_.shape)


def auto_chunk_size(width, dtype, k=0.1):
	if k >= 1.0 or k <= 0:
		raise ValueError("Invalid k argument: should lie between 0 and 1")

	max_ram = k*virtual_memory().available
	bytes_per_val = np.dtype(dtype).itemsize
	chunksize = floor(max_ram / (bytes_per_val * width))
	return chunksize
