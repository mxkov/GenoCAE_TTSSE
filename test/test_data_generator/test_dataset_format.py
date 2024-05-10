import os
import pathlib
import sys
import tensorflow as tf
import tensorflow.distribute as tfd
import tensorflow.distribute.experimental as tfde
from tensorflow.python.distribute.values import PerReplica

GCAE_DIR = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(os.path.join(GCAE_DIR, 'utils'))
from data_handler_distrib import DataGenerator
from distrib_config import get_node_ids, get_worker_id, get_node_roles, define_distribution_strategy


class SampleCounter:

	def __init__(self, fileprefix="samplecount_"):
		self.all_workers   = get_node_ids()
		self.cur_worker_id = get_worker_id()
		assert self.cur_worker_id in self.all_workers

		_, is_chief = get_node_roles()
		self.is_chief = is_chief

		self.fileprefix = fileprefix
		filedir = pathlib.Path(__file__).parent.resolve()

		self.files = []
		for wid in self.all_workers:
			filepath = os.path.join(filedir, self.fileprefix+wid)
			self.files.append(filepath)
			if wid == self.cur_worker_id:
				self.cur_worker_file = filepath

		if os.path.isfile(self.cur_worker_file):
			oldfile_newname = "old." + self.fileprefix + self.cur_worker_id
			oldfile_newname = os.path.join(filedir, oldfile_newname)
			os.rename(self.cur_worker_file, oldfile_newname)

		f = open(self.cur_worker_file, "w")
		f.close()

	def _get_count_from_file(self, filepath_):
		if not os.path.isfile(filepath_):
			raise FileNotFoundError
		f = open(filepath_, "r")
		cur_count_str = f.read().strip()
		f.close()
		if cur_count_str == "":
			return 0
		try:
			cur_count = int(cur_count_str)
		except ValueError:
			raise RuntimeError(f"Invalid sample count in file {filepath_}."+
			                   " Was the file modified?")
		return cur_count

	def _write_count_to_file(self, count_: int):
		f = open(self.cur_worker_file, "w")
		f.write(str(count_))
		f.close()

	def add(self, add_count: int):
		cur_count  = self._get_count_from_file(self.cur_worker_file)
		cur_count += add_count
		self._write_count_to_file(cur_count)

	def _combine(self):
		if not self.is_chief:
			return
		total_count = 0
		for filepath in self.files:
			cur_count = self._get_count_from_file(filepath)
			total_count += cur_count
		self._write_count_to_file(total_count)

	def compute(self, cleanup=True):
		self._combine()
		total_count = self._get_count_from_file(self.cur_worker_file)
		if cleanup:
			self.clean_up()
		return total_count

	def clean_up(self):
		for filepath in self.files:
			if os.path.isfile(filepath):
				os.remove(filepath)


def analyze_dds(dds, dg, batch_size, stop_after=None):

	n_markers  = dg.n_markers
	last_dims  = 2 if dg.missing_mask else 1
	geno_dtype = tf.as_dtype(dg.geno_dtype)

	element_names = ["inputs", "genos", "orig_mask",
	                 "indpop", "last_batch_flag"]
	true_dtypes   = [geno_dtype, geno_dtype, tf.bool,
	                 tf.string, tf.bool]
	err_msg = None
	batch_count = 0
	sample_count_this_worker = 0

	for batch_inputs, batch_genos, batch_orig_mask, \
	    batch_indpop, last_batch_in_chunk in dds:
		# Each of these elements is a PerReplica, not a tensor
		# (unless there's only one device)
		many_replicas = type(batch_inputs)==PerReplica
		if many_replicas:
			# Convert each element to a tuple of tensors
			batch_inputs        = batch_inputs.values
			batch_genos         = batch_genos.values
			batch_orig_mask     = batch_orig_mask.values
			batch_indpop        = batch_indpop.values
			last_batch_in_chunk = last_batch_in_chunk.values
		else:
			# Wrap each element in a tuple of length 1
			# (so we don't have to check many_replicas anymore)
			batch_inputs        = (batch_inputs,)
			batch_genos         = (batch_genos,)
			batch_orig_mask     = (batch_orig_mask,)
			batch_indpop        = (batch_indpop,)
			last_batch_in_chunk = (last_batch_in_chunk,)
		batch_count += 1

		if batch_count == 1:
			n_devices_cur_worker = len(batch_inputs)

		full_batch = [batch_inputs, batch_genos, batch_orig_mask,
		              batch_indpop, last_batch_in_chunk]

		for i,element in enumerate(full_batch):
			if len(element) != n_devices_cur_worker:
				# Pretty sure it should NEVER happen, but just in case
				err_msg = (f"Batch #{batch_count}: " +
				           f"{element_names[i]} missing on some devices")
				return err_msg

		for dev_id in range(n_devices_cur_worker):
			sample_count_this_worker += batch_inputs[dev_id].shape[0]
			if last_batch_in_chunk[dev_id]:
				cur_batch_size = batch_inputs[dev_id].shape[0]
			else:
				cur_batch_size = batch_size
			true_shapes = [(cur_batch_size, n_markers, last_dims), # inputs
			               (cur_batch_size, n_markers),            # genos
			               (cur_batch_size, n_markers),            # orig. mask
			               (cur_batch_size, 2),                    # indpop
			               (1,)]                                   # last batch
			for i,element in enumerate(full_batch):
				cur_shape = element[dev_id].shape
				tru_shape = true_shapes[i]
				if cur_shape != tru_shape:
					err_msg = (f"Wrong {element_names[i]} shape " +
					           f"in batch #{batch_count}: " +
					           f"got {cur_shape}, expected {tru_shape}")
					return err_msg
				cur_dtype = element[dev_id].dtype
				tru_dtype = true_dtypes[i]
				if cur_dtype != tru_dtype:
					err_msg = (f"Wrong {element_names[i]} dtype " +
					           f"in batch #{batch_count}: " +
					           f"got {cur_dtype}, expected {tru_dtype}")
					return err_msg

		if stop_after is not None:
			if batch_count >= stop_after:
				break

	return err_msg, sample_count_this_worker


def test_dataset_format(f_filebase,
                        f_batch_size, f_missing_mask, f_valid_split,
                        f_impute_missing, f_sparsifies,
                        f_norm_opts_flip, f_norm_opts_missval,
                        f_pref_chunk_size, f_shuffle_dataset):

	strat, _, _ = define_distribution_strategy(multiworker_needed=True)

	num_devices = strat.num_replicas_in_sync
	perreplica_batch_size = f_batch_size // num_devices

	dg_args = {
		"filebase"             : f_filebase,
		"global_batch_size"    : f_batch_size,
		"missing_mask"         : f_missing_mask,
		"valid_split"          : f_valid_split,
		"impute_missing"       : f_impute_missing,
		"sparsifies"           : f_sparsifies,
		"normalization_mode"   : "genotypewise01",
		"normalization_options": {"flip"       : f_norm_opts_flip,
		                          "missing_val": f_norm_opts_missval},
		"pref_chunk_size"      : f_pref_chunk_size,
		"batch_shards"         : False,
		"shuffle_dataset"      : f_shuffle_dataset
	}
	dg = DataGenerator(**dg_args)

	#batch_size_per_replica = 1
	#total_batches = dg.n_train_samples // batch_size_per_replica
	# we get empty batches if total_batches % num_devices != 0

	def make_dds(label):
		dds_ = strat.distribute_datasets_from_function(
		              lambda x: dg.create_dataset_from_pq(x, split=label))
		return dds_

	passed = True
	full_err_msg = ""
	true_num_samples = {"train": dg.n_train_samples,
	                    "valid": dg.n_valid_samples}

	for dds_label in ("train", "valid"):
		dds = make_dds(dds_label)
		err, n_samples_this_worker = analyze_dds(dds, dg, perreplica_batch_size)
		if err is not None:
			passed = False
			full_err_msg += f"In {dds_label}: {err}. "

		# Samples might not be equally split between workers.
		# So we need to sum sample_count_this_worker from all workers
		# and compare to the total number of samples.
		sample_counter = SampleCounter()
		sample_counter.add(n_samples_this_worker)
		num_samples = sample_counter.compute(cleanup=True)
		sample_count_mismatch = num_samples!=true_num_samples[dds_label]
		if sample_count_mismatch:
			passed = False
			full_err_msg += (f" In {dds_label}: " +
			                 f"got {num_samples} samples, "
			                 f"expected {true_num_samples[dds_label]} samples.")

	assert passed, full_err_msg
