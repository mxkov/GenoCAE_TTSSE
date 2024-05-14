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


int32_t_MAX = 2**31-1

def open_parquets(filepaths, inds_fam=None, exact_check=False):
	pqds = pq.ParquetDataset(path_or_paths = filepaths,
	                         thrift_string_size_limit = int32_t_MAX,
	                         thrift_container_size_limit = int32_t_MAX,
	                         use_legacy_dataset = False)
	if inds_fam is None:  # no check
		return pqds
	inds_sch = np.array([entry.name for entry in pqds.schema])
	if exact_check:
		fail_cond = ((len(inds_sch) != len(inds_fam)) or
		             (inds_sch != inds_fam).any())
	else:
		present_in_pq  = np.in1d(inds_fam, inds_sch, invert=False)
		present_in_fam = np.in1d(inds_sch, inds_fam, invert=False)
		fail_cond = (inds_sch[present_in_fam] != inds_fam[present_in_pq]).any()
	if fail_cond:
		raise ValueError("Parquet schema inconsistent with FAM files")
	return pqds


@dataclass
class DataGenerator:
	"""docstring""" # TODO: DOCS & comments
	filebase:              str
	global_batch_size:     int
	drop_inds_file:        str  = None
	geno_dtype:            type = np.float32
	missing_mask:          bool = True
	shuffle_dataset:       bool = True
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
	# Parquet reading options
	pref_chunk_size:       int  = None

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

		pqds = open_parquets(filepaths, inds_fam = cur_ind_pop_list[:,0])

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


	def _mask_and_sparsify(self, genos, indpop, genos_shape, *args):

		sparsify = False
		if len(self.sparsifies) > 0:
			# DO NOT use numpy in what's gonna be graph-traced!! only TF
			sparsify_fraction = tf.random.shuffle(self.sparsifies)[0]
			if sparsify_fraction > 0.0:
				sparsify = True

		mask = tf.fill(genos_shape, 1)

		if not sparsify and not self.missing_mask:
			inputs = tf.expand_dims(genos, axis=-1)
			mask   = tf.cast(mask, tf.bool)
			return inputs, genos, mask, indpop, *args
			# this branch should return the same number of elements
			# as the main body.
			# esp bc self.sparsifies may contain both zeros and non-zeros.

		inputs = tf.identity(genos)
		mask   = tf.cast(mask, tf.as_dtype(self.geno_dtype))
		if self.impute_missing:
			orig_mask = tf.identity(mask)
		else:
			delta_m = tf.where(genos != self.missing_val, 1., 0.)
			delta_m = tf.cast(delta_m, mask.dtype)
			orig_mask = delta_m

		if sparsify:
			probs = tf.random.uniform(shape=genos_shape, minval=0, maxval=1)
			delta_s = tf.where(probs < sparsify_fraction, -1., 0.)
			delta_s = tf.cast(delta_s, mask.dtype)
			mask = tf.math.add(mask, delta_s)
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

def auto_chunk_size(width, dtype, k=0.1):
	if k >= 1.0 or k <= 0:
		raise ValueError("Invalid k argument: should lie between 0 and 1")

	max_ram = k*virtual_memory().available
	bytes_per_val = np.dtype(dtype).itemsize
	chunksize = floor(max_ram / (bytes_per_val * width))
	return chunksize
