import os
import pathlib
import sys
import tensorflow as tf
import tensorflow.distribute as tfd
import tensorflow.distribute.experimental as tfde



def analyze_dds(dds, stop_after=None):
	N = 5  # how many elements are in each batch in dds
	batch_shapes     = [None for i in range(N)]
	shape_mismatches = [0 for i in range(N)]

	batch_count = 0
	# iterate over batches in dds

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
	 shape_mismatches_train) = analyze_dds(dds_train)
	(n_batches_valid, batch_shapes_valid,
	 shape_mismatches_valid) = analyze_dds(dds_valid)

	# TODO: check shapes themselves
	# TODO: check data types?
	shapes_match_train = all([x == 0 for x in shape_mismatches_train])
	shapes_match_valid = all([x == 0 for x in shape_mismatches_valid])
	reslist = [shapes_match_train, shapes_match_valid]
	assert all(reslist)
