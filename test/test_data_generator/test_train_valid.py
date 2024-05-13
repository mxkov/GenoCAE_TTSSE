import os
import pathlib
import sys
import tensorflow as tf
import tensorflow.distribute as tfd
import tensorflow.distribute.experimental as tfde
from tensorflow.python.distribute.values import PerReplica
from math import ceil

GCAE_DIR = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(os.path.join(GCAE_DIR, 'utils'))
from data_handler_distrib_exp import DataGenerator
from distrib_config import (
	get_node_ids,
	get_node_roles,
	get_worker_id,
	define_distribution_strategy
)


class SampleCounter:

	def __init__(self, fileprefix="samplecount_"):
		self.all_workers   = get_node_ids()
		self.cur_worker_id = get_worker_id()
		assert self.cur_worker_id in self.all_workers

		_, chief_id = get_node_roles()
		self.is_chief = chief_id==self.cur_worker_id

		self.fileprefix = fileprefix
		filedir = os.path.join(GCAE_DIR, "ae_reports")
		os.makedirs(filedir, exist_ok=True)

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
			#raise FileNotFoundError(f"{filepath_}"")
			print(f"Warning: {filepath_} not found")
			return 0
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

	def add(self, add_count: int):
		cur_count  = self._get_count_from_file(self.cur_worker_file)
		cur_count += add_count
		f = open(self.cur_worker_file, "w")
		f.write(str(cur_count))
		f.close()

	def _combine(self):
		if not self.is_chief:
			return
		total_count = 0
		for filepath in self.files:
			cur_count = self._get_count_from_file(filepath)
			total_count += cur_count
		for filepath in self.files:
			f = open(filepath, "w")
			f.write(str(total_count))
			f.close()

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


def inspect_batch(n_markers, last_dims, geno_dtype,
                  batch_size, batch_count, *batch_items):

	element_names = ["inputs", "genos", "orig_mask",
	                 "indpop", "last_batch_flag"]
	true_dtypes   = [geno_dtype, geno_dtype, tf.bool,
	                 tf.string, tf.bool]
	err_msgs = []
	sample_count_this_worker = 0

	full_batch = list(batch_items)
	# Each of these elements is a PerReplica, not a tensor
	# (unless there's only one device)

	pr_checks = [type(item)==PerReplica for item in batch_items]
	one_replica   = not any(pr_checks)
	many_replicas = all(pr_checks)
	assert one_replica or many_replicas

	if many_replicas:
		# Convert each element to a tuple of tensors
		full_batch = [item.values for item in full_batch]
	else:
		# Wrap each element in a tuple of length 1
		# (so we don't have to check many_replicas anymore)
		full_batch = [(item,) for item in full_batch]

	n_devices_cur_worker = len(full_batch[0])

	for i,element in enumerate(full_batch):
		if len(element) != n_devices_cur_worker:
			# Pretty sure it should NEVER happen, but just in case
			err_msgs.append(f"Batch #{batch_count}: " +
			                f"{element_names[i]} missing on some devices.")
			#return err_msgs

	for dev_id in range(n_devices_cur_worker):
		sample_count_this_worker += full_batch[0][dev_id].shape[0]
		last_batch_in_chunk = full_batch[-1][dev_id]
		if last_batch_in_chunk:
			cur_batch_size = full_batch[0][dev_id].shape[0]
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
				err_msgs.append(f"Wrong {element_names[i]} shape " +
				                f"in batch #{batch_count}: " +
				                f"got {cur_shape}, expected {tru_shape}.")
				#return err_msgs
			cur_dtype = element[dev_id].dtype
			tru_dtype = true_dtypes[i]
			if cur_dtype != tru_dtype:
				err_msgs.append(f"Wrong {element_names[i]} dtype " +
				                f"in batch #{batch_count}: " +
				                f"got {cur_dtype}, expected {tru_dtype}.")
				#return err_msgs

	return err_msgs, sample_count_this_worker


def test_train_valid(f_filebase,
                     f_batch_size, f_missing_mask, f_valid_split,
                     f_impute_missing, f_sparsifies,
                     f_norm_opts_flip, f_norm_opts_missval,
                     f_pref_chunk_size, f_shuffle_dataset,
                     f_valid_every, f_stop_after):

	strat, _, _ = define_distribution_strategy(multiworker_needed=True)
	num_devices = strat.num_replicas_in_sync
	print(f"\nDEVICES: {num_devices}\n")
	global_batch_size = f_batch_size * num_devices

	dg_args = {
		"filebase"             : f_filebase,
		"global_batch_size"    : global_batch_size,
		"missing_mask"         : f_missing_mask,
		"valid_split"          : f_valid_split,
		"impute_missing"       : f_impute_missing,
		"sparsifies"           : f_sparsifies,
		"normalization_mode"   : "genotypewise01",
		"normalization_options": {"flip"       : f_norm_opts_flip,
		                          "missing_val": f_norm_opts_missval},
		"pref_chunk_size"      : f_pref_chunk_size,
		"batch_shards"         : False,
		"shuffle_dataset"      : f_shuffle_dataset,
		"_debug"               : True
	}
	dg = DataGenerator(**dg_args)

	#batch_size_per_replica = 1
	#total_batches = dg.n_train_samples // batch_size_per_replica
	# we get empty batches if total_batches % num_devices != 0

	n_markers  = dg.n_markers
	last_dims  = 2 if dg.missing_mask else 1
	geno_dtype = tf.as_dtype(dg.geno_dtype)

	def make_dds(label):
		dds_ = strat.distribute_datasets_from_function(
		              lambda x: dg.create_dataset_from_pq(x, split=label))
		return dds_

	dds_train = make_dds("train")
	dds_valid = make_dds("valid")

	batch_count_train = 0
	batch_count_valid = None
	train_samples_this_worker = 0
	valid_samples_this_worker = None

	samples_passed = 0
	validations = 0
	valid_mismatches_b = []
	valid_mismatches_s = []
	errs_train = []
	errs_valid = []

	for batch_elems in dds_train:
		batch_count_train += 1
		errs, n_samples = inspect_batch(n_markers, last_dims, geno_dtype,
		                                f_batch_size,
		                                batch_count_train,
		                                *batch_elems)
		errs_train.extend(errs)
		train_samples_this_worker += n_samples
		samples_passed += global_batch_size

		if f_valid_every is not None and samples_passed >= f_valid_every:
			samples_passed = 0
			validations += 1
			cur_batch_count_valid = 0
			cur_valid_samples = 0
			for valid_batch_elems in dds_valid:
				cur_batch_count_valid += 1
				errs, n_samples = inspect_batch(n_markers, last_dims, geno_dtype,
				                                f_batch_size,
				                                cur_batch_count_valid,
				                                *valid_batch_elems)
				errs_valid.extend(errs)
				cur_valid_samples += n_samples

			if batch_count_valid is None:
				batch_count_valid = cur_batch_count_valid
			if batch_count_valid != cur_batch_count_valid:
				err = (f"Valid #{validations}: got {cur_batch_count_valid}, "+
				       f"expected {batch_count_valid}")
				valid_mismatches_b.append(err)

			if valid_samples_this_worker is None:
				valid_samples_this_worker = cur_valid_samples
			if valid_samples_this_worker != cur_valid_samples:
				err = (f"Valid #{validations}: got {cur_valid_samples}, "+
				       f"expected {valid_samples_this_worker}")
				valid_mismatches_s.append(err)

		if f_stop_after is not None and batch_count_train >= f_stop_after:
			break

	# that One More Valid Block
	if ceil(samples_passed / global_batch_size) > 5:
		samples_passed = 0
		validations += 1
		cur_batch_count_valid = 0
		cur_valid_samples = 0
		for valid_batch_elems in dds_valid:
			cur_batch_count_valid += 1
			errs, n_samples = inspect_batch(n_markers, last_dims, geno_dtype,
			                                f_batch_size,
			                                cur_batch_count_valid,
			                                *valid_batch_elems)
			errs_valid.extend(errs)
			cur_valid_samples += n_samples
		batch_count_valid = cur_batch_count_valid
		valid_samples_this_worker = cur_valid_samples

	wid = get_worker_id()
	print(f"\nWORKER STATS FOR {wid} (valid_every = {f_valid_every}):\n")
	print(f"Total train samples: {train_samples_this_worker}")
	print(f"Total valid samples: {valid_samples_this_worker}\n")
	print(f"Train batches: {batch_count_train}")
	print(f"Valid batches: {batch_count_valid}\n")
	print(f"Total validation runs: {validations}")
	print(f"Valid sample mismatches: {len(valid_mismatches_s)}")
	print(f"Valid  batch mismatches: {len(valid_mismatches_b)}\n")
	print(f"Train errors: {len(errs_train)}")
	print(f"Valid errors: {len(errs_valid)}\n")



if __name__ == "__main__":

	fixt_params = {
		"f_filebase"         : "/mnt/ukb/ukb22418_b0_v2",
		"f_batch_size"       : 40,
		"f_missing_mask"     : True,
		"f_valid_split"      : 0.02,
		"f_impute_missing"   : False,
		"f_sparsifies"       : [0.0, 0.1, 0.2, 0.3, 0.4],
		"f_norm_opts_flip"   : False,
		"f_norm_opts_missval": -1,
		"f_pref_chunk_size"  : 7440,
		"f_shuffle_dataset"  : True,
		"f_valid_every"      : 10000,  # None to disable
		"f_stop_after"       : None
	}

	test_train_valid(**fixt_params)
