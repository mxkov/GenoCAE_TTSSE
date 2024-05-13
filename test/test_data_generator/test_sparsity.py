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
from distrib_config import get_worker_id, define_distribution_strategy

## Motivation for this test:
## "Stateful dataset transformations are currently not supported with tf.distribute 
##  and any stateful ops that the dataset may have are currently ignored".
##  https://www.tensorflow.org/tutorials/distribute/input#caveats
##  (as of TF ver. 2.15)


def check_dds_sparsity(dds, dg, label, epoch,
                       stop_after=None, verbose_=False):

	missval = dg.missing_val
	wid = get_worker_id()
	err_msg = None
	batch_count = 0

	if verbose_:
		print(f"\n{label.upper()}, epoch {epoch}")

	for batch_inputs, batch_genos, batch_orig_mask, _, _ in dds:
		# Each of these elements is a PerReplica, not a tensor
		# (unless there's only one device)
		many_replicas = type(batch_inputs)==PerReplica
		if many_replicas:
			# Convert each element to a tuple of tensors
			batch_inputs    = batch_inputs.values
			batch_genos     = batch_genos.values
			batch_orig_mask = batch_orig_mask.values
		else:
			# Wrap each element in a tuple of length 1
			# (so we don't have to check many_replicas anymore)
			batch_inputs    = (batch_inputs,)
			batch_genos     = (batch_genos,)
			batch_orig_mask = (batch_orig_mask,)
		batch_count += 1

		if batch_count == 1:
			n_devices_cur_worker = len(batch_inputs)

		for dev_id in range(n_devices_cur_worker):

			inputs = batch_inputs[dev_id][:,:,0]
			mask   = tf.cast(batch_inputs[dev_id][:,:,1], tf.bool)
			genos  = batch_genos[dev_id]
			if inputs.shape != genos.shape:
				err_msg = (f"Shape mismatch between inputs and genos " +
				           f"in batch #{batch_count}: " +
				           f"{inputs.shape} vs {genos.shape}")
				return err_msg

			genotypes_match = tf.reduce_all(tf.equal(inputs[mask], genos[mask]))
			if not genotypes_match:
				err_msg = (f"{wid}: batch #{batch_count}:" +
				           " Preserved input genotypes" +
				           " do not match true genotypes")
				return err_msg

			pos_sparse  = tf.math.not_equal(inputs, genos)
			pos_missing = inputs==missval
			sparsified = tf.reduce_all((pos_sparse & pos_missing) == pos_sparse)
			#
			gsize = genos.shape[0]*genos.shape[1]
			sparse_frac = tf.math.count_nonzero(pos_sparse).numpy() / gsize
			if verbose_:
				print(f"{wid}, batch #{batch_count}:" +
				      " sparse frac {:.6f}".format(sparse_frac))
			if not sparsified:
				err_msg = f"{wid}: batch #{batch_count} was not sparsified"
				return err_msg

		if stop_after is not None:
			if batch_count >= stop_after:
				break

	return err_msg


def test_sparsity(f_filebase,
                  f_batch_size, f_missing_mask, f_valid_split,
                  f_impute_missing, f_sparsifies,
                  f_norm_opts_flip, f_norm_opts_missval,
                  f_pref_chunk_size, f_shuffle_dataset,
                  epochs=1, verbose=False):

	strat, _, _ = define_distribution_strategy(multiworker_needed=True)

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

	def make_dds(label):
		dds_ = strat.distribute_datasets_from_function(
		              lambda x: dg.create_dataset_from_pq(x, split=label))
		return dds_

	passed = True
	full_err_msg = ""

	for dds_label in ("train", "valid"):
		dds = make_dds(dds_label)
		for e in range(1, epochs+1):
			err = check_dds_sparsity(dds, dg, dds_label, e, verbose_=verbose)
			if err is not None:
				passed = False
				full_err_msg += f"In {dds_label}, epoch {e}: {err}. "

	assert passed, full_err_msg


if __name__ == "__main__":
	fixt_params_tiny = {
		"f_filebase"         : os.path.join("example_tiny", "parquet",
		                                     "HumanOrigins249_tiny"),
		"f_batch_size"       : 12,
		"f_missing_mask"     : True,
		"f_valid_split"      : 0.2,
		"f_impute_missing"   : False,
		"f_sparsifies"       : [0.0, 0.1, 0.2, 0.3, 0.4],
		"f_norm_opts_flip"   : False,
		"f_norm_opts_missval": -1,
		"f_pref_chunk_size"  : None,
		"f_shuffle_dataset"  : True,
		"epochs"             : 3,
		"verbose"            : True
	}
	test_sparsity(**fixt_params_tiny)
