import os
import pathlib
import sys
import tensorflow as tf
import tensorflow.distribute as tfd
import tensorflow.distribute.experimental as tfde
from tensorflow.types.experimental.distributed import PerReplica



def analyze_dds(dds, dg, batch_size, stop_after=None):

	n_markers  = dg.n_markers
	last_dims  = 2 if dg.missing_mask else 1
	geno_dtype = tf.as_dtype(dg.geno_dtype)

	true_dtypes = [geno_dtype for i in range(3)] + [tf.string, tf.bool]

	err_msg = None
	batch_count = 0

	element_names = ["inputs", "genos", "orig_mask",
	                 "indpop", "last_batch_flag"]
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

		for j in range(n_devices_cur_worker):
			if last_batch_in_chunk[j]:
				cur_batch_size = batch_inputs[j].shape[0]
			else:
				cur_batch_size = batch_size
			true_shapes = [(cur_batch_size, n_markers, last_dims),
			               (cur_batch_size, n_markers),
			               (cur_batch_size, n_markers),
			               (cur_batch_size, 2),
			               (1,)]
			for i,element in enumerate(full_batch):
				cur_shape = element[j].shape
				tru_shape = true_shapes[i]
				if cur_shape != tru_shape:
					err_msg = (f"Wrong {element_names[i]} shape " +
					           f"in batch #{batch_count}: " +
					           f"got {cur_shape}, expected {tru_shape}")
					return err_msg
				cur_dtype = element[j].dtype
				tru_dtype = true_dtypes[i]
				if cur_dtype != tru_dtype:
					err_msg = (f"Wrong {element_names[i]} dtype " +
					           f"in batch #{batch_count}: " +
					           f"got {cur_dtype}, expected {tru_dtype}")
					return err_msg

		if stop_after is not None:
			if batch_count >= stop_after:
				break

	return err_msg


def test_dataset_format():

	GCAE_DIR = pathlib.Path(__file__).resolve().parents[1]
	sys.path.append(os.path.join(GCAE_DIR, 'utils'))
	from data_handler_distrib import data_generator_distrib
	from tf_config import set_tf_config

	if "SLURMD_NODENAME" in os.environ:
		num_workers, chief_id = set_tf_config()
		resolver  = tfd.cluster_resolver.TFConfigClusterResolver()
		comm_opts = tfde.CommunicationOptions(
		                   implementation=tfde.CommunicationImplementation.NCCL)
		# TODO: this doesn't work atm.
		#       to try: tfde strat? comm other than NCCL?
		strat = tfd.MultiWorkerMirroredStrategy(cluster_resolver=resolver,
		                                        communication_options=comm_opts)
	else:
		num_workers = 1
		chief_id = None
		strat = tfd.MirroredStrategy()

	num_devices = strat.num_replicas_in_sync

	# TODO: make fixtures for diff param combinations
	batch_size = 30
	dg_args = {
		"filebase"             : "sometestdata", # TODO
		"global_batch_size"    : batch_size * num_devices,
		"tfrecords_prefix"     : "",
		"missing_mask"         : True,
		"valid_split"          : 0.2,
		"impute_missing"       : False,
		"normalization_mode"   : "genotypewise01",
		"normalization_options": {"flip": False, "missing_val": -1.0},
		"sparsifies"           : [0.0, 0.1, 0.2, 0.3, 0.4],
		"pref_chunk_size"      : None,
		"batch_shards"         : False,
		"shuffle_dataset"      : True
	}
	dg = data_generator_distrib(**dg_args)
	
	def make_dds(label):
		dds = strat.distribute_datasets_from_function(
		              lambda x: dg.create_dataset_from_pq(x, split=label))
		#n_batches = get_dds_size(dds) # <- takes forever
		return dds

	dds_train = make_dds("train")
	dds_valid = make_dds("valid")

	err_train = analyze_dds(dds_train, dg, batch_size)
	err_valid = analyze_dds(dds_valid, dg, batch_size)

	full_err_msg = ""
	if err_train is not None:
		full_err_msg += f" In train: {err_train}."
	if err_valid is not None:
		full_err_msg += f" In valid: {err_valid}."

	assert err_train is None and err_valid is None, full_err_msg
