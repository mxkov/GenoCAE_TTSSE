import os
import pathlib
import sys
import tensorflow as tf
import tensorflow.distribute as tfd
import tensorflow.distribute.experimental as tfde
from tensorflow.types.experimental.distributed import PerReplica



def analyze_dds(dds, stop_after=None):
	N = 5  # how many elements are in each batch in dds
	batch_shapes     = [None for i in range(N)]
	shape_mismatches = [0 for i in range(N)]
	n_chunks         = []
	last_batch_shapes= []

	batch_count = 0
	for batch_inputs, batch_genos, batch_orig_mask, \
	    batch_indpop, last_batch_in_chunk in dds:
		# Each of these is a PerReplica, not a tensor
		# (unless there's only one device)
		full_batch = [batch_inputs, batch_genos, batch_orig_mask,
		              batch_indpop, last_batch_in_chunk]
		many_replicas = type(batch_inputs)==PerReplica
		if many_replicas:
			full_batch = [x.values for x in full_batch]
			# Now they are tuples of tensors
		batch_count += 1

		if batch_count == 1:
			n_devices_cur_worker = len(full_batch[0]) if many_replicas else 1
			n_chunks = [0 for j in range(n_devices_cur_worker)]
			last_batch_shapes = [[] for j in range(n_devices_cur_worker)]
			for i in range(N):
				if many_replicas:
					batch_shapes[i] = full_batch[i][0].shape
				else:
					batch_shapes[i] = full_batch[i].shape

		# If any replica has a last batch in chunk,
		# record its shape and skip to the next batch
		if many_replicas:
			last_batch_where = [j for j in range(n_devices_cur_worker)
			                    if last_batch_in_chunk.values[j]==True]
			if len(last_batch_where) > 0:
				for j_ in last_batch_where:
					last_batch_shapes[j_].append([x[j_].shape
					                              for x in full_batch])
					n_chunks[j_] += 1
				continue
		else:
			if last_batch_in_chunk:
				last_batch_shapes[0] = [x.shape for x in full_batch]
				n_chunks[0] += 1
				continue

		for i in range(N):
			if many_replicas:
				cur_element_shape = full_batch[i][0].shape
				# TODO: how to make these checks if not many_replicas?
				assert len(full_batch[i]) == n_devices_cur_worker
				for j in range(1,n_devices_cur_worker):
					assert full_batch[i][j].shape == cur_element_shape
			else:
				cur_element_shape = full_batch[i].shape
			if batch_shapes[i] != cur_element_shape:
				shape_mismatches[i] += 1
			# TODO: check these shapes in other ways (e.g. # of markers)

		if stop_after is not None:
			if batch_count >= stop_after:
				break

	return batch_count, batch_shapes, shape_mismatches


def test_dataset_format():

	GCAE_DIR = pathlib.Path(__file__).resolve().parents[1]
	sys.path.append(os.path.join(GCAE_DIR, 'utils'))
	from data_handler_distrib import data_generator_distrib
	from tf_config import set_tf_config

	num_workers, chief_id = set_tf_config()
	strat = tfd.MirroredStrategy()

	num_devices = strat.num_replicas_in_sync

	dg_args = {
		# TODO
	}
	dg = data_generator_distrib(**dg_args)
	
	def make_dds(label):
		dds = strat.distribute_datasets_from_function(
		              lambda x: dg.create_dataset_from_pq(x, split=label))
		#n_batches = get_dds_size(dds) # <- takes forever
		return dds

	dds_train = make_dds("train")
	dds_valid = make_dds("valid")

	(n_batches_train, batch_shapes_train,
	 shape_mismatches_train) = analyze_dds(dds_train, num_devices)
	(n_batches_valid, batch_shapes_valid,
	 shape_mismatches_valid) = analyze_dds(dds_valid, num_devices)

	# TODO: check shapes themselves
	# TODO: check data types?
	shapes_match_train = all([x == 0 for x in shape_mismatches_train])
	shapes_match_valid = all([x == 0 for x in shape_mismatches_valid])
	reslist = [shapes_match_train, shapes_match_valid]
	assert all(reslist)
