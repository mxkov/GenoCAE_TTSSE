"""GenoCAE.

Usage:
  run_gcae.py train --datadir=<name> --data=<name> --model_id=<name> --train_opts_id=<name> --data_opts_id=<name> --epochs=<num> [--resume_from=<num> --trainedmodeldir=<name> --patience=<num> --save_interval=<num> --start_saving_from=<num> ]
  run_gcae.py project --datadir=<name>   [ --data=<name> --model_id=<name>  --train_opts_id=<name> --data_opts_id=<name> --superpops=<name> --epoch=<num> --trainedmodeldir=<name>   --pdata=<name> --trainedmodelname=<name>]
  run_gcae.py plot --datadir=<name> [  --data=<name>  --model_id=<name> --train_opts_id=<name> --data_opts_id=<name>  --superpops=<name> --epoch=<num> --trainedmodeldir=<name>  --pdata=<name> --trainedmodelname=<name>]
  run_gcae.py animate --datadir=<name>   [ --data=<name>   --model_id=<name> --train_opts_id=<name> --data_opts_id=<name>  --superpops=<name> --epoch=<num> --trainedmodeldir=<name> --pdata=<name> --trainedmodelname=<name>]
  run_gcae.py evaluate --datadir=<name> --metrics=<name>  [  --data=<name>  --model_id=<name> --train_opts_id=<name> --data_opts_id=<name>  --superpops=<name> --epoch=<num> --trainedmodeldir=<name>  --pdata=<name> --trainedmodelname=<name>]

Options:
  -h --help                  show this screen
  --datadir=<name>           directory where sample data is stored. if not absolute: assumed relative to GenoCAE/ directory. DEFAULT: data/
  --data=<name>              file prefix, not including path, of the data files (EIGENSTRAT of PLINK format)
  --trainedmodeldir=<name>   base path where to save model training directories. if not absolute: assumed relative to GenoCAE/ directory. DEFAULT: ae_out/
  --model_id=<name>          model id, corresponding to a file models/{model_id}.json
  --train_opts_id=<name>     train options id, corresponding to a file train_opts/{train_opts_id}.json
  --data_opts_id=<name>      data options id, corresponding to a file data_opts/{data_opts_id}.json
  --epochs<num>              number of epochs to train
  --resume_from<num>         saved epoch to resume training from. set to -1 for latest saved epoch. DEFAULT: None (don't resume)
  --save_interval<num>       epoch intervals at which to save state of model. DEFAULT: None (don't save)
  --start_saving_from<num>   number of epochs to train before starting to save model state. DEFAULT: 0.
  --trainedmodelname=<name>  name of the model training directory to fetch saved model state from when project/plot/evaluating
  --pdata=<name>             file prefix, not including path, of data to project/plot/evaluate. if not specified, assumed to be the same the model was trained on.
  --epoch<num>               epoch at which to project/plot/evaluate data. DEFAULT: all saved epochs
  --superpops<name>          path+filename of file mapping populations to superpopulations. used to color populations of the same superpopulation in similar colors in plotting. if not absolute path: assumed relative to GenoCAE/ directory.
  --metrics=<name>           the metric(s) to evaluate, e.g. hull_error of f1 score. can pass a list with multiple metrics, e.g. "f1_score_3,f1_score_5". DEFAULT: f1_score_3
  --patience=<num>           stop training after this number of epochs without improving lowest validation. DEFAULT: None

"""

# TODO next:
# https://www.tensorflow.org/guide/distributed_training
# https://www.tensorflow.org/tutorials/distribute/multi_worker_with_ctl
# https://www.tensorflow.org/tutorials/distribute/save_and_load
# https://www.tensorflow.org/tutorials/distribute/input

from docopt import docopt, DocoptExit
import tensorflow as tf
import tensorflow.distribute as tfd
import tensorflow.distribute.experimental as tfde
from tensorflow.keras import Model, layers
from tensorflow.python.training.server_lib import ClusterSpec
from datetime import datetime
from utils.data_handler_distrib import data_generator_distrib # TODO: look into reimplementing all these below as well:
from utils.data_handler import get_saved_epochs, get_projected_epochs, write_h5, read_h5, get_coords_by_pop, convex_hull_error, f1_score_kNN, plot_genotype_hist, to_genotypes_sigmoid_round, to_genotypes_invscale_round, GenotypeConcordance, get_pops_with_k, get_ind_pop_list_from_map, get_baseline_gc, write_metric_per_epoch_to_csv
from utils.visualization import plot_coords_by_superpop, plot_clusters_by_superpop, plot_coords, plot_coords_by_pop, make_animation, write_f1_scores_to_csv
from utils.tf_config import set_tf_config
import utils.visualization
import utils.layers
import json
import numpy as np
import time
import os
import glob
import math
import matplotlib.pyplot as plt
import shutil
import csv
import copy
import h5py
import matplotlib.animation as animation
from pathlib import Path

# TODO: add PerReplica checks!!!

def _isChief():
	if "isChief" in os.environ:
		return os.environ["isChief"] == "1"
	return True

def chief_print1(msg):
	if "isChief" in os.environ:
		if os.environ["isChief"] == "1":
			print(msg)
	else:
		print(msg)


class SlurmClusterResolver_fixed(tfd.cluster_resolver.SlurmClusterResolver):
	"""Child of tf.distribute.cluster_resolver.SlurmClusterResolver
	that does NOT assume GPU availablity."""

	def cluster_spec(self):
		"""Returns a ClusterSpec object based on the latest instance group info.

		This returns a ClusterSpec object for use based on information from the
		specified initialization parameters and Slurm environment variables. The
		cluster specification is resolved each time this function is called. The
		resolver extract hostnames of nodes by scontrol and pack tasks in that
		order until a node a has number of tasks that is equal to specification.
		GPUs on nodes are allocated to tasks by specification through setting
		CUDA_VISIBLE_DEVICES environment variable.

		Returns:
		  A ClusterSpec containing host information retrieved from Slurm's
		    environment variables.
		"""

		task_list = []
		self._gpu_allocation = []
		self._cluster_allocation = {}

		# Sort to make sure the order is the same for each run
		for host, num_tasks in sorted(self._task_configuration.items()):
			# num_tasks is per node
			if self._gpus_per_node == 0 or self._gpus_per_task == 0:
				gpu_starting_ids_this_node = [None for _ in range(num_tasks)]
			else:
				gpu_starting_ids_this_node = range(0, self._gpus_per_node,
				                                      self._gpus_per_task)

			for port_offset, gpu_offset in zip(range(num_tasks),
			                                   gpu_starting_ids_this_node):
				host_addr = '%s:%d' % (host, self._port_base + port_offset)
				task_list.append(host_addr)

				if gpu_offset is None:
					self._gpu_allocation.append('')
					continue
				gpu_id_list = []
				for gpu_id in range(gpu_offset, gpu_offset+self._gpus_per_task):
					gpu_id_list.append(str(gpu_id))
				self._gpu_allocation.append(','.join(gpu_id_list))

		cluster_rank_offset_start = 0
		cluster_rank_offset_end = 0

		# Sort to make sure the order is the same for each run
		for task_type, num_tasks in sorted(self._jobs.items()):
			cluster_rank_offset_end = cluster_rank_offset_start + num_tasks

			self._cluster_allocation[task_type] = (
			       task_list[cluster_rank_offset_start:cluster_rank_offset_end])

			if cluster_rank_offset_start <= self._rank < cluster_rank_offset_end:
				self.task_type = task_type
				self.task_id = self._rank - cluster_rank_offset_start

			cluster_rank_offset_start = cluster_rank_offset_end

		if self._auto_set_gpu:
			os.environ['CUDA_VISIBLE_DEVICES']=self._gpu_allocation[self._rank]

		return ClusterSpec(self._cluster_allocation)


GCAE_DIR = Path(__file__).resolve().parent
class Autoencoder(Model):

	def __init__(self, model_architecture, n_markers, noise_std, regularizer):
		'''

		Initiate the autoencoder with the specified options.
		All variables of the model are defined here.

		:param model_architecture: dict containing a list of layer representations
		:param n_markers: number of markers / SNPs in the data
		:param noise_std: standard deviation of noise to add to encoding layer during training. False if no noise.
		:param regularizer: dict containing regularizer info. False if no regularizer.
		'''
		super(Autoencoder, self).__init__()
		self.all_layers = []
		self.n_markers = n_markers
		self.noise_std = noise_std
		self.residuals = dict()
		self.marker_spec_var = False

		chief_print("\n______________________________ Building model ______________________________")
		# variable that keeps track of the size of layers in encoder, to be used when constructing decoder.
		ns=[]
		ns.append(n_markers)

		first_layer_def = model_architecture["layers"][0]
		layer_module = getattr(eval(first_layer_def["module"]), first_layer_def["class"])
		layer_args = first_layer_def["args"]
		try:
			activation = getattr(tf.nn, layer_args["activation"])
			layer_args.pop("activation")
			first_layer = layer_module(activation=activation, **layer_args)

		except KeyError:
			first_layer = layer_module(**layer_args)
			activation = None

		self.all_layers.append(first_layer)
		chief_print("Adding layer: " + str(layer_module.__name__) + ": " + str(layer_args))

		if first_layer_def["class"] == "conv1d" and "strides" in layer_args.keys() and layer_args["strides"] > 1:
			ns.append(int(first_layer.shape[1]))

		# add all layers except first
		for layer_def in model_architecture["layers"][1:]:
			layer_module = getattr(eval(layer_def["module"]), layer_def["class"])
			layer_args = layer_def["args"]

			for arg in ["size", "layers", "units", "shape", "target_shape", "output_shape", "kernel_size", "strides"]:

				if arg in layer_args.keys():
					layer_args[arg] = eval(str(layer_args[arg]))

			if layer_def["class"] == "MaxPool1D":
				ns.append(int(math.ceil(float(ns[-1]) / layer_args["strides"])))

			if layer_def["class"] == "Conv1D" and "strides" in layer_def.keys() and layer_def["strides"] > 1:
				raise NotImplementedError

			chief_print("Adding layer: " + str(layer_module.__name__) + ": " + str(layer_args))

			if "name" in layer_args and (layer_args["name"] == "i_msvar" or layer_args["name"] == "nms"):
				self.marker_spec_var = True

			if "activation" in layer_args.keys():
				activation = getattr(tf.nn, layer_args["activation"])
				layer_args.pop("activation")
				this_layer = layer_module(activation=activation, **layer_args)
			else:
				this_layer = layer_module(**layer_args)

			self.all_layers.append(this_layer)

		if noise_std:
			self.noise_layer = tf.keras.layers.GaussianNoise(noise_std)

		self.ns = ns
		self.regularizer = regularizer

		if self.marker_spec_var:
			random_uniform = tf.random_uniform_initializer()
			self.ms_variable = tf.Variable(random_uniform(shape = (1, n_markers), dtype=tf.float32), name="marker_spec_var")
			self.nms_variable = tf.Variable(random_uniform(shape = (1, n_markers), dtype=tf.float32), name="nmarker_spec_var")
		else:
			chief_print("No marker specific variable.")


	def call(self, input_data, is_training = True, verbose = False):
		'''
		The forward pass of the model. Given inputs, calculate the output of the model.

		:param input_data: input data
		:param is_training: if called during training
		:param verbose: print the layers and their shapes
		:return: output of the model (last layer) and latent representation (encoding layer)

		'''

		# if we're adding a marker specific variables as an additional channel
		if self.marker_spec_var:
			# Tiling it to handle the batch dimension

			ms_tiled = tf.tile(self.ms_variable, (tf.shape(input_data)[0], 1))
			ms_tiled = tf.expand_dims(ms_tiled, 2)
			nms_tiled = tf.tile(self.nms_variable, (tf.shape(input_data)[0], 1))
			nms_tiled = tf.expand_dims(nms_tiled, 2)
			concatted_input = tf.concat([input_data, ms_tiled], 2)
			input_data = concatted_input

		if verbose:
			chief_print("inputs shape " + str(input_data.shape))

		first_layer = self.all_layers[0]
		counter = 1

		if verbose:
			chief_print("layer {0}".format(counter))
			chief_print("--- type: {0}".format(type(first_layer)))

		x = first_layer(inputs=input_data)

		if "Residual" in first_layer.name:
			out = self.handle_residual_layer(first_layer.name, x, verbose=verbose)
			if not out == None:
				x = out
		if verbose:
			chief_print("--- shape: {0}".format(x.shape))

		# indicator if were doing genetic clustering (ADMIXTURE-style) or not
		have_encoded_raw = False

		# initializing encoded data
		encoded_data = None

		# do all layers except first
		for layer_def in self.all_layers[1:]:
			try:
				layer_name = layer_def.cname
			except:
				layer_name = layer_def.name

			counter += 1

			if verbose:
				chief_print("layer {0}: {1} ({2}) ".format(counter, layer_name, type(layer_def)))

			if layer_name == "dropout":
				x = layer_def(x, training = is_training)
			else:
				x = layer_def(x)

			# If this is a clustering model then we add noise to the layer first in this step
			# and the next layer, which is sigmoid, is the actual encoding.
			if layer_name == "encoded_raw":
				have_encoded_raw = True
				if self.noise_std:
					x = self.noise_layer(x, training = is_training)
				encoded_data_raw = x

			# If this is the encoding layer, we add noise if we are training
			elif "encoded" in layer_name:
				if self.noise_std and not have_encoded_raw:
					x = self.noise_layer(x, training = is_training)
				encoded_data = x

			if "Residual" in layer_name:
				out = self.handle_residual_layer(layer_name, x, verbose=verbose)
				if not out == None:
					x = out

			# inject marker-specific variable by concatenation
			if "i_msvar" in layer_name and self.marker_spec_var:
				x = self.injectms(verbose, x, layer_name, ms_tiled, self.ms_variable)

			if "nms" in layer_name and self.marker_spec_var:
				x = self.injectms(verbose, x, layer_name, nms_tiled, self.nms_variable)

			if verbose:
				chief_print("--- shape: {0}".format(x.shape))

		if self.regularizer and encoded_data is not None:
			reg_module = eval(self.regularizer["module"])
			reg_name = getattr(reg_module, self.regularizer["class"])
			reg_func = reg_name(float(self.regularizer["reg_factor"]))

			# if this is a clustering model then the regularization is added to the raw encoding, not the softmaxed one
			if have_encoded_raw:
				reg_loss = reg_func(encoded_data_raw)
			else:
				reg_loss = reg_func(encoded_data)
			self.add_loss(reg_loss)

		return x, encoded_data


	def handle_residual_layer(self, layer_name, input, verbose=False):
		suffix = layer_name.split("Residual_")[-1].split("_")[0]
		res_number = suffix[0:-1]
		if suffix.endswith("a"):
			if verbose:
				chief_print("encoder-to-decoder residual: saving residual {}".format(res_number))
			self.residuals[res_number] = input
			return None
		if suffix.endswith("b"):
			if verbose:
				chief_print("encoder-to-decoder residual: adding residual {}".format(res_number))
			residual_tensor = self.residuals[res_number]
			res_length = residual_tensor.shape[1]
			if len(residual_tensor.shape) == 3:
				x = tf.keras.layers.Add()([input[:,0:res_length,:], residual_tensor])
			if len(residual_tensor.shape) == 2:
				x = tf.keras.layers.Add()([input[:,0:res_length], residual_tensor])

			return x

	def injectms(self, verbose, x, layer_name, ms_tiled, ms_variable):
		if verbose:
			chief_print("----- injecting marker-specific variable")

		# if we need to reshape ms_variable before concatting it
		if not self.n_markers == x.shape[1]:
			d = int(math.ceil(float(self.n_markers) / int(x.shape[1])))
			diff = d*int(x.shape[1]) - self.n_markers
			ms_var = tf.reshape(tf.pad(ms_variable,[[0,0],[0,diff]]), (-1, x.shape[1],d))
			# Tiling it to handle the batch dimension
			ms_tiled = tf.tile(ms_var, (tf.shape(x)[0],1,1))

		else:
			# Tiling it to handle the batch dimension
			ms_tiled = tf.tile(ms_variable, (x.shape[0],1))
			ms_tiled = tf.expand_dims(ms_tiled, 2)

		if "_sg" in layer_name:
			if verbose:
				chief_print("----- stopping gradient for marker-specific variable")
			ms_tiled = tf.stop_gradient(ms_tiled)


		if verbose:
			chief_print("ms var {}".format(ms_variable.shape))
			chief_print("ms tiled {}".format(ms_tiled.shape))
			chief_print("concatting: {0} {1}".format(x.shape, ms_tiled.shape))

		x = tf.concat([x, ms_tiled], 2)


		return x

@tf.function
def run_optimization(model, optimizer, loss_function,
                     input, targets, mask):
	'''
	Run one step of optimization process based on the given data.

	:param model: a tf.keras.Model
	:param optimizer: a tf.keras.optimizers
	:param loss_function: a loss function
	:param input: input data
	:param targets: target data
	:return: value of the loss function
	'''
	with tf.GradientTape() as g:
		output, encoded_data = model(input, is_training=True)
		loss_value = loss_function(y_pred = output, y_true = targets,
		                           orig_nonmissing_mask = mask,
		                           model_losses = model.losses)

	gradients = g.gradient(loss_value, model.trainable_variables)

	optimizer.apply_gradients(zip(gradients, model.trainable_variables),
	                         #skip_gradients_aggregation=True)        # new Adam
	                          experimental_aggregate_gradients=False) # legacy Adam
	# TODO: "If true, gradients aggregation will not be performed inside optimizer.
	#        Usually this arg is set to True when you write custom code
	#        aggregating gradients outside the optimizer. "
	#        ...maybe for phenomodel?
	return loss_value

@tf.function
def train_step_distrib(model_, optimizer_, loss_function_,
                       input_, targets_, mask_):

	per_replica_losses = strat.run(run_optimization,
	                               args=(model_, optimizer_, loss_function_,
	                                     input_, targets_, mask_))
	loss = strat.reduce("SUM", per_replica_losses, axis=None)

	return loss 

@tf.function
def compute_loss(model, loss_function, input, targets, mask):
	output, _  = model(input, is_training=False)
	loss_value = loss_function(y_pred = output, y_true = targets,
	                           orig_nonmissing_mask = mask,
	                           model_losses = model.losses)
	return loss_value

@tf.function
def compute_loss_distrib(model_, loss_function_, input_, targets_, mask_):
	per_replica_losses = strat.run(compute_loss,
	                               args=(model_, loss_function_,
	                                     input_, targets_, mask_))
	loss = strat.reduce("SUM", per_replica_losses, axis=None)
	return loss


def get_batches(n_samples, batch_size):
	n_batches = n_samples // batch_size

	n_samples_last_batch = n_samples % batch_size
	if n_samples_last_batch > 0:
		n_batches += 1
	else:
		n_samples_last_batch = batch_size

	return n_batches, n_samples_last_batch

def alfreqvector(y_pred):
	'''
	Get a probability distribution over genotypes from y_pred.
	Assumes y_pred is raw model output, one scalar value per genotype.

	Scales this to (0,1) and interprets this as a allele frequency, uses formula
	for Hardy-Weinberg equilibrium to get probabilities for genotypes [0,1,2].

	:param y_pred: (n_samples x n_markers) tensor of raw network output for each sample and site
	:return: (n_samples x n_markers x 3 tensor) of genotype probabilities for each sample and site
	'''

	if len(y_pred.shape) == 2:
		alfreq = tf.keras.activations.sigmoid(y_pred)
		alfreq = tf.expand_dims(alfreq, -1)
		return tf.concat(((1-alfreq) ** 2, 2 * alfreq * (1 - alfreq), alfreq ** 2), axis=-1)
	else:
		return tf.nn.softmax(y_pred)


class WeightKeeper:
	# "Apparently, in order to save the model, the save_model call needs to be made on all processes,
	# but they cannot be saved to the same file, since that causes a race condition".
	# TODO: maybe hide get_saved_epochs() in here.

	def __init__(self, train_directory, default_prefix=""):
		chief_weights_dirname = "weights"
		if _isChief():
			weights_dirname = chief_weights_dirname
		else:
			weights_dirname = "weights_tmp_" + os.environ["SLURMD_NODENAME"]
		weights_dir = os.path.join(train_directory, weights_dirname)
		if not os.path.isdir(weights_dir):
			os.makedirs(weights_dir)
		weights_dir_chief = os.path.join(train_directory, chief_weights_dirname)

		self.train_dir = train_directory
		self.weights_dir = weights_dir
		self.weights_dir_chief = weights_dir_chief
		self.default_prefix = default_prefix

	def get_full_fileprefix(self, epoch_, prefix=None, chief_only=True):
		if prefix is None:
			prefix = self.default_prefix
		wd = self.weights_dir_chief if chief_only else self.weights_dir
		return os.path.join(wd, f"{prefix}{epoch_}")

	def save(self, model, epoch, update_best=False):
		if update_best:
			new_prefix = "min_valid."
			prev_best_prefix = self.get_full_fileprefix("*", prefix=new_prefix,
			                                            chief_only=False)
			for f in glob.glob(prev_best_prefix):
				os.remove(f)
		else:
			new_prefix = self.default_prefix
		weights_file_prefix = self.get_full_fileprefix(epoch, prefix=new_prefix,
		                                               chief_only=False)
		startTime = datetime.now()
		model.save_weights(weights_file_prefix, save_format ="tf")
		save_time = (datetime.now() - startTime).total_seconds()
		chief_print(f"-------- Saving weights: {weights_file_prefix} "+
		            f"time: {save_time}")

	def cleanup(self):
		if not _isChief():
			assert self.weights_dir != self.weights_dir_chief
			shutil.rmtree(self.weights_dir)



if __name__ == "__main__":
	chief_print = print
	chief_print(f"\n{datetime.now().time()}\n")
	chief_print("tensorflow version {0}".format(tf.__version__))
	tf.keras.backend.set_floatx('float32')

	try:
		arguments = docopt(__doc__, version='GenoCAE 1.1.0')
	except DocoptExit:
		chief_print("Invalid command. Run 'python run_gcae.py --help' for more information.")
		exit(1)

	for k in list(arguments.keys()):
		knew = k.split('--')[-1]
		arg=arguments.pop(k)
		arguments[knew]=arg

	gpus = tf.config.list_physical_devices(device_type="GPU")
	chief_print("Available GPU devices:\n{}".format(gpus))
	num_physical_gpus = len(gpus)
	gpus = ["gpu:"+ str(i) for i in range(num_physical_gpus)]


	## Define distribution strategies

	if "SLURMD_NODENAME" in os.environ:

		slurm_job = 1
		num_workers, chief_id = set_tf_config()
		isChief = os.environ["SLURMD_NODENAME"] == chief_id
		os.environ["isChief"] = json.dumps(int(isChief))
		chief_print("Number of workers: {}".format(num_workers))

		if num_workers > 1 and arguments["train"]:
			#resolver = tfd.cluster_resolver.TFConfigClusterResolver()
			resolver=SlurmClusterResolver_fixed(gpus_per_node=num_physical_gpus)
			if num_physical_gpus > 0:
				comm_impl = tfde.CommunicationImplementation.NCCL
				os.environ["NCCL_DEBUG"] = "INFO" # or "WARN"
			else:
				comm_impl = tfde.CommunicationImplementation.RING
			#comm_impl = tfde.CommunicationImplementation.AUTO
			comm_opts = tfde.CommunicationOptions(implementation = comm_impl)
			# CollectiveCommunication is deprecated in TF 2.7
			strat = tfd.MultiWorkerMirroredStrategy(cluster_resolver = resolver,
			                                        communication_options = comm_opts)

		else:
			if not isChief:
				print("Work has ended for this worker")
				exit(0)
			slurm_job = 0
			# TODO: make it not nccl when no gpus?
			strat = tfd.MirroredStrategy(devices = gpus,
			                             cross_device_ops = tfd.NcclAllReduce())

	else:
		isChief = True
		os.environ["isChief"] = json.dumps(int(isChief))
		slurm_job = 0
		num_workers = 1
		strat = tfd.MirroredStrategy()

	num_devices = strat.num_replicas_in_sync
	chief_print('Number of devices: {}'.format(num_devices))


	if arguments["trainedmodeldir"]:
		trainedmodeldir = arguments["trainedmodeldir"]
		if not os.path.isabs(trainedmodeldir):
			trainedmodeldir = os.path.join(GCAE_DIR, trainedmodeldir)

	else:
		trainedmodeldir = os.path.join(GCAE_DIR, "ae_out")

	if arguments["datadir"]:
		datadir = arguments["datadir"]
		if not os.path.isabs(datadir):
			datadir = os.path.join(GCAE_DIR, datadir)

	else:
		datadir = os.path.join(GCAE_DIR, "data")

	if arguments["trainedmodelname"]:
		trainedmodelname = arguments["trainedmodelname"]
		train_directory = os.path.join(trainedmodeldir, trainedmodelname)

		namesplit = trainedmodelname.split(".")
		data_opts_id = namesplit[3]
		train_opts_id = namesplit[2]
		model_id = namesplit[1]
		data = namesplit[4]

	else:
		data = arguments["data"]
		data_opts_id = arguments["data_opts_id"]
		train_opts_id = arguments["train_opts_id"]
		model_id = arguments["model_id"]
		train_directory = False

	with open(os.path.join(GCAE_DIR, "data_opts", data_opts_id+".json")) as data_opts_def_file:
		data_opts = json.load(data_opts_def_file)

	with open(os.path.join(GCAE_DIR, "train_opts", train_opts_id+".json")) as train_opts_def_file:
		train_opts = json.load(train_opts_def_file)

	with open(os.path.join(GCAE_DIR, "models", model_id+".json")) as model_def_file:
		model_architecture = json.load(model_def_file)

	for layer_def in model_architecture["layers"]:
		if "args" in layer_def.keys() and "name" in layer_def["args"].keys() and "encoded" in layer_def["args"]["name"] and "units" in layer_def["args"].keys():
			n_latent_dim = layer_def["args"]["units"]

	# indicator of whether this is a genetic clustering or dimensionality reduction model
	doing_clustering = False
	for layer_def in model_architecture["layers"][1:-1]:
		if "encoding_raw" in layer_def.keys():
			doing_clustering = True

	chief_print("\n______________________________ arguments ______________________________")
	for k in arguments.keys():
		chief_print(k + " : " + str(arguments[k]))
	chief_print("\n______________________________ data opts ______________________________")
	for k in data_opts.keys():
		chief_print(k + " : " + str(data_opts[k]))
	chief_print("\n______________________________ train opts ______________________________")
	for k in train_opts:
		chief_print(k + " : " + str(train_opts[k]))
	chief_print("______________________________")


	batch_size = train_opts["batch_size"]
	global_batch_size = train_opts["batch_size"] * num_devices
	learning_rate = train_opts["learning_rate"] * num_devices
	# TODO: ^ check this later
	regularizer = train_opts["regularizer"]

	try:
		max_batches = int(train_opts["max_batches"])
	except:
		max_batches = None

	superpopulations_file = arguments['superpops']
	if superpopulations_file and not os.path.isabs(os.path.dirname(superpopulations_file)):
		superpopulations_file = os.path.join(GCAE_DIR,
		                                     os.path.dirname(superpopulations_file),
		                                     Path(superpopulations_file).name)

	norm_opts = data_opts["norm_opts"]
	norm_mode = data_opts["norm_mode"]
	missing_mask_input = data_opts["missing_mask"]
	validation_split = data_opts["validation_split"]

	if "sparsifies" in data_opts.keys():
		sparsify_input = True
		n_input_channels = 2
		sparsifies = data_opts["sparsifies"]

	else:
		sparsify_input = False
		n_input_channels = 1
		sparsifies = []

	if "impute_missing" in data_opts.keys():
		fill_missing = data_opts["impute_missing"]

	else:
		fill_missing = False

	if fill_missing:
		chief_print("Imputing originally missing genotypes to most common value.")
	else:
		chief_print("Keeping originally missing genotypes.")
		n_input_channels = 2

	if not train_directory:
		train_directory = os.path.join(trainedmodeldir,
		                               ".".join(("ae", model_id,
		                                         train_opts_id, data_opts_id,
		                                         data)))

	if arguments["pdata"]:
		pdata = arguments["pdata"]
	else:
		pdata = data

	data_prefix = os.path.join(datadir, pdata)
	results_directory = os.path.join(train_directory, pdata)
	try:
		os.makedirs(results_directory)
	except OSError:
		pass

	encoded_data_file = os.path.join(train_directory, pdata, "encoded_data.h5")

	if "noise_std" in train_opts.keys():
		noise_std = train_opts["noise_std"]
	else:
		noise_std = False

	if (arguments['evaluate'] or arguments['animate'] or arguments['plot']):

		if os.path.isfile(encoded_data_file):
			encoded_data = h5py.File(encoded_data_file, 'r')
		else:
			chief_print("------------------------------------------------------------------------")
			chief_print("Error: File {0} not found.".format(encoded_data_file))
			chief_print("------------------------------------------------------------------------")
			exit(1)

		epochs = get_projected_epochs(encoded_data_file)

		if arguments['epoch']:
			epoch = int(arguments['epoch'])
			if epoch in epochs:
				epochs = [epoch]
			else:
				chief_print("------------------------------------------------------------------------")
				chief_print("Error: Epoch {0} not found in {1}.".format(epoch, encoded_data_file))
				chief_print("------------------------------------------------------------------------")
				exit(1)

		if doing_clustering:
			if arguments['animate']:
				chief_print("------------------------------------------------------------------------")
				chief_print("Error: Animate not supported for genetic clustering model.")
				chief_print("------------------------------------------------------------------------")
				exit(1)


			if arguments['plot'] and not superpopulations_file:
				chief_print("------------------------------------------------------------------------")
				chief_print("Error: Plotting of genetic clustering results requires a superpopulations file.")
				chief_print("------------------------------------------------------------------------")
				exit(1)

	else:

		loss_def = train_opts["loss"]

		loss_class = getattr(eval(loss_def["module"]), loss_def["class"])
		if "args" in loss_def.keys():
			loss_args = loss_def["args"]
		else:
			loss_args = dict()
		loss_args["reduction"] = tf.keras.losses.Reduction.NONE

	if arguments['train']:

		ae_weight_manager = WeightKeeper(train_directory)
		epochs = int(arguments["epochs"])

		try:
			save_interval = int(arguments["save_interval"])
		except:
			save_interval = epochs

		try:
			start_saving_from = int(arguments["start_saving_from"])
		except:
			start_saving_from = 0

		try:
			patience = int(arguments["patience"])
		except:
			patience = epochs

		try:
			resume_from = int(arguments["resume_from"])
			if resume_from < 1:
				saved_weights_dir = ae_weight_manager.weights_dir_chief
				saved_epochs = get_saved_epochs(saved_weights_dir)
				resume_from = saved_epochs[-1]
		except:
			resume_from = False

		dg_args = {"filebase"             : data_prefix,
		           "global_batch_size"    : global_batch_size,
		           "missing_mask"         : missing_mask_input,
		           "valid_split"          : validation_split,
		           "impute_missing"       : fill_missing,
		           "normalization_mode"   : norm_mode,
		           "normalization_options": norm_opts,
		           "sparsifies"           : sparsifies,
		           "pref_chunk_size"      : None, # auto
		           "shuffle_dataset"      : True}
		dg = data_generator_distrib(**dg_args)
		n_markers = copy.deepcopy(dg.n_markers)

		# dds stands for tfd.DistributedDataset
		dds_train = strat.distribute_datasets_from_function(
		                  lambda x: dg.create_dataset_from_pq(x, split="train"))
		if validation_split > 0.0: # TODO: mb another condition
			dds_valid = strat.distribute_datasets_from_function(
			              lambda x: dg.create_dataset_from_pq(x, split="valid"))
		# TODO: right now, valid set is defined in this method.
		#       if it gets moved to init or outer scope, we can call these later,
		#       after setting up lr schedule and such.
		chief_print(f"Data chunk size: {dg.pref_chunk_size}")

		n_unique_train_samples = copy.deepcopy(dg.n_train_samples)
		n_valid_samples = copy.deepcopy(dg.n_valid_samples)

		# TODO: i don't understand what this commented block is for,
		#       will prob remove or reimplement later
		# TODO: upd: it's for when we want to only take a fraction of train samples.
		#       should be done at the level of data generator though.
		#       as an arg to dataset creation.
		#if "n_samples" in train_opts.keys() and int(train_opts["n_samples"]) > 0:
		#	n_train_samples = int(train_opts["n_samples"])
		#else:
		#	n_train_samples = n_unique_train_samples
		n_train_samples = copy.deepcopy(n_unique_train_samples)

		global_batch_size_valid = global_batch_size
		batch_size_valid = batch_size
		# TODO: double-check if these batch sizes are consistent with dg.
		#       maybe we can even put get_batches() in there as a method.
		#       google, how to get number of batches in a tf.data.Dataset obj?
		# TODO: actually, since get_batches() is only used by the scheduler,
		#       there should be an easy way to do without.
		#       for example, save batch count to wherever we're resuming from.
		(n_train_batches,
		 n_train_samples_last_batch) = get_batches(n_train_samples, batch_size)
		(n_valid_batches,
		 n_valid_samples_last_batch) = get_batches(n_valid_samples, batch_size_valid)

		train_times  = []
		train_epochs = []
		save_epochs  = []

		############### setup learning rate schedule ##############
		step_counter = resume_from * n_train_batches
		if "lr_scheme" in train_opts.keys():
			schedule_module = getattr(eval(train_opts["lr_scheme"]["module"]), train_opts["lr_scheme"]["class"])
			schedule_args = train_opts["lr_scheme"]["args"]

			if "decay_every" in schedule_args:
				decay_every = int(schedule_args.pop("decay_every"))
				decay_steps = n_train_batches * decay_every
				schedule_args["decay_steps"] = decay_steps

			lr_schedule = schedule_module(learning_rate, **schedule_args)

			# use the schedule to calculate what the lr was at the epoch were resuming from
			updated_lr = lr_schedule(step_counter)
			lr_schedule = schedule_module(updated_lr, **schedule_args)

			chief_print("Using learning rate schedule {0}.{1} with {2}".format(train_opts["lr_scheme"]["module"], train_opts["lr_scheme"]["class"], schedule_args))
		else:
			lr_schedule = False

		chief_print("\n______________________________ Data ______________________________")
		chief_print("N unique train samples: {0}".format(n_unique_train_samples))
		chief_print("--- training on : {0}".format(n_train_samples))
		chief_print("N valid samples: {0}".format(n_valid_samples))
		chief_print("N markers: {0}".format(n_markers))
		chief_print("")

		with strat.scope():
			autoencoder = Autoencoder(model_architecture, n_markers,
			                          noise_std, regularizer)
			# In newer TF versions, Adam doesn't have _decayed_lr() anymore.
			#optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule)
			# Temporary fix: legacy Adam
			optimizer = tf.keras.optimizers.legacy.Adam(learning_rate = lr_schedule)
			loss_obj = loss_class(**loss_args)

			@tf.function
			def loss_func(y_pred, y_true, orig_nonmissing_mask, model_losses):
				y_pred = y_pred[:, 0:n_markers]
				y_pred = alfreqvector(y_pred)
				y_true = tf.one_hot(tf.cast(y_true*2, tf.uint8), 3)*0.9997 + 0.0001
				
				#if not fill_missing and y_pred.shape[0] != 0:
				# The shape check is a crutch for empty batches:
				# if one of the replicas receives a batch with 0 samples,
				# the mask will have shape (0,0) not matching y_pred and y_true.
				# TODO: handle this better.
				# empty batches STILL produce errors in other places,
				# e.g. PredictLoss(). Handle them BEFORE they get there.
				if not fill_missing:
					y_pred = y_pred[orig_nonmissing_mask]
					y_true = y_true[orig_nonmissing_mask]
				
				per_example_loss  = loss_obj(y_pred=y_pred, y_true=y_true)
				#per_example_loss /= self.n_markers  # TODO: ask why
				loss = tf.nn.compute_average_loss(per_example_loss)
				loss += tf.nn.scale_regularization_loss(sum(model_losses))
				return loss

		# get a single batch to:
		# a) run through optimization to reload weights and optimizer variables,
		# b) print layer dims
		(input_init, targets_init,
		    orig_mask_init, _, _) = next(iter(dds_train))
		# get the same for every device
		if num_devices > 1: # TODO: replace with PerReplica check
			input_init     = input_init.values[0]
			targets_init   = targets_init.values[0]
			orig_mask_init = orig_mask_init.values[0]

		if resume_from:
			chief_print("\n______________________________ Resuming training from epoch {0} ______________________________".format(resume_from))
			weights_file_prefix = ae_weight_manager.get_full_fileprefix(
			                                          str(resume_from))
			chief_print(f"Reading weights from {weights_file_prefix}")

			# This initializes the variables used by the optimizers,
			# as well as any stateful metric variables
			_ = train_step_distrib(autoencoder, optimizer, loss_func,
			                       input_init[0,:], targets_init[0,:],
			                       orig_mask_init[0,:])
			with strat.scope():
				autoencoder.load_weights(weights_file_prefix)

		chief_print("\n______________________________ Train ______________________________")

		# a small run-through of the model with just 2 samples for printing the dimensions of the layers (verbose=True)
		chief_print("Model layers and dimensions:")
		chief_print("-----------------------------")
		with strat.scope():
			_, _ = autoencoder(input_init[:2,:], is_training=False, verbose=True)

		######### Create objects for tensorboard summary ###############################

		if isChief: # TODO: make sure only chief can write to it
			twdir = os.path.join(train_directory, "train")
			vwdir = os.path.join(train_directory, "valid")
			train_writer = tf.summary.create_file_writer(twdir)
			valid_writer = tf.summary.create_file_writer(vwdir)

		######################################################

		# train losses per epoch
		losses_t = []
		# valid losses per epoch
		losses_v = []
		# min loss stats
		min_valid_loss = np.inf
		min_valid_loss_epoch = None

		chief_print(f"\nTraining start: {datetime.now().time()}")

		for e in range(1,epochs+1):
			# TODO: profiler, mayhaps?
			#       https://www.tensorflow.org/guide/profiler
			startTime = datetime.now()
			effective_epoch = e + resume_from
			losses_t_batches = []
			losses_v_batches = []
			batch_count_train = 0
			batch_count_valid = 0

			for batch_input, batch_target, batch_orig_mask, _, _ in dds_train:
				train_batch_loss = train_step_distrib(autoencoder,
				                                      optimizer, loss_func,
				                                      batch_input, batch_target,
				                                      batch_orig_mask)
				losses_t_batches.append(train_batch_loss)
				step_counter += 1
				batch_count_train += 1
				if max_batches is not None and batch_count_train > max_batches:
					break
			train_loss_this_epoch = np.average(losses_t_batches)

			train_time = (datetime.now() - startTime).total_seconds()
			train_times.append(train_time)
			train_epochs.append(effective_epoch)
			losses_t.append(train_loss_this_epoch)

			if isChief:
				with train_writer.as_default():
					tf.summary.scalar("loss", train_loss_this_epoch,
 					                  step=step_counter)
					if lr_schedule:
						tf.summary.scalar("learning_rate",
						       optimizer._decayed_lr(var_dtype=tf.float32),
						       step=step_counter)
					else:
						tf.summary.scalar("learning_rate", learning_rate,
						                  step=step_counter)

			chief_print("")
			chief_print("Epoch: {}/{}...".format(effective_epoch, epochs+resume_from))
			chief_print("--- Train loss: {:.4f}  time: {} ({})".format(
			          train_loss_this_epoch, train_time, datetime.now().time()))

			if n_valid_samples > 0:

				startTime = datetime.now()

				for batch_input_valid, batch_target_valid,\
				    batch_orig_mask, _, _ in dds_valid:
					valid_loss_batch = compute_loss_distrib(autoencoder,
					                                        loss_func,
					                                        batch_input_valid,
					                                        batch_target_valid,
					                                        batch_orig_mask)
					losses_v_batches.append(valid_loss_batch)
					batch_count_valid += 1
					if max_batches is not None and batch_count_valid > max_batches:
						break
				valid_loss_this_epoch = np.average(losses_v_batches)

				valid_time = (datetime.now() - startTime).total_seconds()
				losses_v.append(valid_loss_this_epoch)

				if isChief:
					with valid_writer.as_default():
						tf.summary.scalar("loss", valid_loss_this_epoch,
						                  step=step_counter)

				if valid_loss_this_epoch <= min_valid_loss:
					min_valid_loss = valid_loss_this_epoch
					prev_min_val_loss_epoch = min_valid_loss_epoch
					min_valid_loss_epoch = effective_epoch

					if e > start_saving_from:
						ae_weight_manager.save(autoencoder, effective_epoch,
						                       update_best=True)

				evals_since_min_valid_loss = effective_epoch - min_valid_loss_epoch
				chief_print("--- Valid loss: {:.4f}  time: {} ({}) min loss: {:.4f} epochs since: {}".format(
				       valid_loss_this_epoch, valid_time, datetime.now().time(),
				       min_valid_loss, evals_since_min_valid_loss))

				if evals_since_min_valid_loss >= patience:
					break

			if e % save_interval == 0 and e > start_saving_from :
				ae_weight_manager.save(autoencoder, effective_epoch)

		ae_weight_manager.save(autoencoder, effective_epoch)

		if isChief:
			outfilename = os.path.join(train_directory, "train_times.csv")
			write_metric_per_epoch_to_csv(outfilename, train_times, train_epochs)
	
			outfilename = os.path.join(train_directory, "losses_from_train_t.csv")
			epochs_t_combined, losses_t_combined = write_metric_per_epoch_to_csv(outfilename, losses_t, train_epochs)
			fig, ax = plt.subplots()
			plt.plot(epochs_t_combined, losses_t_combined, label="train", c="orange")
	
			if n_valid_samples > 0:
				outfilename = os.path.join(train_directory, "losses_from_train_v.csv")
				epochs_v_combined, losses_v_combined = write_metric_per_epoch_to_csv(outfilename, losses_v, train_epochs)
				plt.plot(epochs_v_combined, losses_v_combined, label="valid", c="blue")
				min_valid_loss_epoch = epochs_v_combined[np.argmin(losses_v_combined)]
				plt.axvline(min_valid_loss_epoch, color="black")
				plt.text(min_valid_loss_epoch + 0.1, 0.5,'min valid loss at epoch {}'.format(int(min_valid_loss_epoch)),
						 rotation=90,
						 transform=ax.get_xaxis_text1_transform(0)[0])
	
			plt.xlabel("Epoch")
			plt.ylabel("Loss function value")
			plt.legend()
			plt.savefig(os.path.join(train_directory, "losses_from_train.pdf"))
			plt.close()

		chief_print("Done training at {0}. Wrote to {1}".format(
	             datetime.now().time(), train_directory))
		ae_weight_manager.cleanup()

	if arguments['project']:
		# TODO: the 'train' suffix in this section is confusing.
		if not isChief:
			print("Work has ended for this worker")
			exit(0)
		# TODO: implement multi-worker projecting someday

		ae_weight_manager = WeightKeeper(train_directory)
		projected_epochs = get_projected_epochs(encoded_data_file)

		if arguments['epoch']:
			epoch = int(arguments['epoch'])
			epochs = [epoch]

		else:
			saved_weights_dir = ae_weight_manager.weights_dir_chief
			epochs = get_saved_epochs(saved_weights_dir)

		for projected_epoch in projected_epochs:
			try:
				epochs.remove(projected_epoch)
			except:
				continue

		chief_print("Projecting epochs: {0}".format(epochs))
		chief_print("Already projected: {0}".format(projected_epochs))

		dg_args = {"filebase"             : data_prefix,
		           "global_batch_size"    : global_batch_size,
		           "missing_mask"         : missing_mask_input,
		           "valid_split"          : 0.0,
		           "impute_missing"       : fill_missing,
		           "normalization_mode"   : norm_mode,
		           "normalization_options": norm_opts,
		           "sparsifies"           : [0.0],
		           "pref_chunk_size"      : None, # auto
		           "shuffle_dataset"      : False}
		dg = data_generator_distrib(**dg_args)
		n_markers = copy.deepcopy(dg.n_markers)

		# dds stands for tfd.DistributedDataset
		dds_proj = strat.distribute_datasets_from_function(
		                    lambda x: dg.create_dataset_from_pq(x, split="all"))

		# TODO: we don't need strat scope in this section,
		#       we terminated all non-chief workers already...
		#       also global batch size is too much
		# TODO: actually we do need the strat.
		#       so we can properly finalize the dataset.
		#       that was avoidable, but whoops, too late.
		genotype_concordance_metric = GenotypeConcordance()
		with strat.scope():
			autoencoder = Autoencoder(model_architecture, n_markers,
			                          noise_std, regularizer)
			loss_obj = loss_class(**loss_args)

			@tf.function
			def loss_func(y_pred, y_true, orig_nonmissing_mask, model_losses):
				y_pred = y_pred[:, 0:n_markers]
				y_pred = alfreqvector(y_pred)
				y_true = tf.one_hot(tf.cast(y_true*2, tf.uint8), 3)*0.9997 + 0.0001
				
				if not fill_missing:
					y_pred = y_pred[orig_nonmissing_mask]
					y_true = y_true[orig_nonmissing_mask]
				
				per_example_loss  = loss_obj(y_pred=y_pred, y_true=y_true)
				#per_example_loss /= self.n_markers  # TODO: ask why
				loss = tf.nn.compute_average_loss(per_example_loss)
				loss += tf.nn.scale_regularization_loss(sum(model_losses))
				return loss

		# loss function of the train set per epoch
		losses_train = []

		# genotype concordance of the train set per epoch
		genotype_concs_train = []

		scatter_points_per_epoch = []
		colors_per_epoch = []
		markers_per_epoch = []
		edgecolors_per_epoch = []

		for epoch in epochs:
			chief_print("########################### epoch {0} ###########################".format(epoch))
			weights_file_prefix=ae_weight_manager.get_full_fileprefix(str(epoch))
			chief_print(f"Reading weights from {weights_file_prefix}")

			# TODO: find out if we should run 1-2 samples through optimization
			#       in order to load the weights here.
			with strat.scope():
				autoencoder.load_weights(weights_file_prefix)

			ind_pop_list_train = np.empty(shape=(0,2), dtype=str)
			encoded_train = np.empty((0, n_latent_dim))
			decoded_train = None
			targets_train = np.empty((0, n_markers))
			orig_mask_train = np.empty((0, n_markers))
			# TODO: all these arrays might outgrow RAM. do something.

			loss_value_per_batch = []
			genotype_conc_per_batch = []

			for batch_input, batch_target,\
			    batch_orig_mask, batch_indpop, _ in dds_proj:

				decoded_batch, encoded_batch = autoencoder(batch_input,
				                                           is_training=False)
				loss_batch = loss_func(y_pred=decoded_batch,
				                       y_true=batch_target,
				                       orig_nonmissing_mask=batch_orig_mask,
				                       model_losses=autoencoder.losses)

				ind_pop_list_train=np.concatenate((ind_pop_list_train, batch_indpop), axis=0)
				encoded_train=np.concatenate((encoded_train, encoded_batch), axis=0)
				if decoded_train is None:
					decoded_train=np.copy(decoded_batch[:,0:n_markers])
				else:
					decoded_train=np.concatenate((decoded_train, decoded_batch[:,0:n_markers]), axis=0)
				targets_train=np.concatenate((targets_train, batch_target[:,0:n_markers]), axis=0)

				orig_mask_train = np.concatenate((orig_mask_train,
				                                  batch_orig_mask), axis=0)

				loss_value_per_batch.append(loss_batch)

			ind_pop_list_train = np.array(ind_pop_list_train)
			encoded_train = np.array(encoded_train)
			orig_mask_train = tf.cast(orig_mask_train, tf.bool)
			loss_value = np.average(loss_value_per_batch)

			if epoch == epochs[0]:
				assert len(ind_pop_list_train) == dg.n_total_samples, \
				       f"{len(ind_pop_list_train)} vs {dg.n_total_samples}"
				assert len(encoded_train) == dg.n_total_samples, \
				       f"{len(encoded_train)} vs {dg.n_total_samples}"

				write_h5(encoded_data_file, "ind_pop_list_train",
				         np.array(ind_pop_list_train, dtype='S'))

			genotype_concordance_metric.reset_states()

			# TODO: maybe copy train_opts to project_opts in this mode or something. just for clarity.
			if train_opts["loss"]["class"] == "MeanSquaredError" and (data_opts["norm_mode"] == "smartPCAstyle" or data_opts["norm_mode"] == "standard"):
				try:
					scaler = dg.scaler
				except:
					chief_print("Could not calculate predicted genotypes and genotype concordance. No scaler available in data handler.")
					genotypes_output = np.array([])
					true_genotypes = np.array([])

				genotypes_output = to_genotypes_invscale_round(decoded_train[:, 0:n_markers], scaler_vals = scaler)
				true_genotypes = to_genotypes_invscale_round(targets_train, scaler_vals = scaler)
				genotype_concordance_metric.update_state(y_pred = genotypes_output[orig_mask_train],
				                                         y_true = true_genotypes[orig_mask_train])


			elif train_opts["loss"]["class"] == "BinaryCrossentropy" and data_opts["norm_mode"] == "genotypewise01":
				genotypes_output = to_genotypes_sigmoid_round(decoded_train[:, 0:n_markers])
				true_genotypes = targets_train
				genotype_concordance_metric.update_state(y_pred = genotypes_output[orig_mask_train],
				                                         y_true = true_genotypes[orig_mask_train])

			elif train_opts["loss"]["class"] in ["CategoricalCrossentropy", "KLDivergence"] and data_opts["norm_mode"] == "genotypewise01":
				genotypes_output = tf.cast(tf.argmax(alfreqvector(decoded_train[:, 0:n_markers]), axis = -1), tf.float16) * 0.5
				true_genotypes = targets_train
				genotype_concordance_metric.update_state(y_pred = genotypes_output[orig_mask_train],
				                                         y_true = true_genotypes[orig_mask_train])

			else:
				chief_print("Could not calculate predicted genotypes and genotype concordance. Not implemented for loss {0} and normalization {1}.".format(train_opts["loss"]["class"], data_opts["norm_mode"]))
				genotypes_output = np.array([])
				true_genotypes = np.array([])

			genotype_concordance_value = genotype_concordance_metric.result()

			losses_train.append(loss_value)
			genotype_concs_train.append(genotype_concordance_value)

			if superpopulations_file:
				# TODO: might have to reimplement all these functions from the old data handler,
				#       to make them work with the new one.
				coords_by_pop = get_coords_by_pop(data_prefix, encoded_train, ind_pop_list = ind_pop_list_train)

				if doing_clustering:
					plot_clusters_by_superpop(coords_by_pop,
					                          os.path.join(results_directory,
					                                       f"clusters_e_{epoch}"),
					                          superpopulations_file,
					                          write_legend = epoch == epochs[0])
				else:
					scatter_points, colors, markers, edgecolors = \
						plot_coords_by_superpop(coords_by_pop,
						                        os.path.join(results_directory,
						                                     f"dimred_e_{epoch}_by_superpop"),
						                        superpopulations_file,
						                        plot_legend = epoch == epochs[0])

					scatter_points_per_epoch.append(scatter_points)
					colors_per_epoch.append(colors)
					markers_per_epoch.append(markers)
					edgecolors_per_epoch.append(edgecolors)

			else:
				try:
					coords_by_pop = get_coords_by_pop(data_prefix, encoded_train, ind_pop_list = ind_pop_list_train)
					plot_coords_by_pop(coords_by_pop,
					                   os.path.join(results_directory,
					                                f"dimred_e_{epoch}_by_pop"))
				except:
					plot_coords(encoded_train,
					            os.path.join(results_directory,
					                         f"dimred_e_{epoch}"))

			write_h5(encoded_data_file, f"{epoch}_encoded_train", encoded_train)

		try:
			plot_genotype_hist(np.array(genotypes_output),
			                   os.path.join(results_directory,
			                                f"output_as_genotypes_e{epoch}"))
			plot_genotype_hist(np.array(true_genotypes),
			                   os.path.join(results_directory,
			                                "true_genotypes"))
		except:
			pass

		############################### losses ##############################

		outfilename = os.path.join(results_directory, "losses_from_project.csv")
		epochs_combined, losses_train_combined = write_metric_per_epoch_to_csv(outfilename, losses_train, epochs)

		plt.plot(epochs_combined, losses_train_combined,
		         label="all data", c="red")
		plt.xlabel("Epoch")
		plt.ylabel("Loss function value")
		plt.legend()
		plt.savefig(os.path.join(results_directory, "losses_from_project.pdf"))
		plt.close()


		############################### gconc ###############################
		try:
			baseline_genotype_concordance = get_baseline_gc(true_genotypes)
		except:
			baseline_genotype_concordance = None

		outfilename = os.path.join(results_directory, "genotype_concordances.csv")
		epochs_combined, genotype_concs_combined = write_metric_per_epoch_to_csv(outfilename, genotype_concs_train, epochs)

		plt.plot(epochs_combined, genotype_concs_combined, label="train", c="orange")
		if baseline_genotype_concordance:
			plt.plot([epochs_combined[0], epochs_combined[-1]], [baseline_genotype_concordance, baseline_genotype_concordance], label="baseline", c="black")

		plt.xlabel("Epoch")
		plt.ylabel("Genotype concordance")

		plt.savefig(os.path.join(results_directory, "genotype_concordances.pdf"))

		plt.close()

	if arguments['animate']:
		if not isChief:
			print("Work has ended for this worker")
			exit(0)

		chief_print("Animating epochs {}".format(epochs))

		FFMpegWriter = animation.writers['ffmpeg']
		scatter_points_per_epoch = []
		colors_per_epoch = []
		markers_per_epoch = []
		edgecolors_per_epoch = []

		ind_pop_list_train = read_h5(encoded_data_file, "ind_pop_list_train")

		for epoch in epochs:
			chief_print("########################### epoch {0} ###########################".format(epoch))

			encoded_train = read_h5(encoded_data_file, "{0}_encoded_train".format(epoch))

			coords_by_pop = get_coords_by_pop(data_prefix, encoded_train, ind_pop_list = ind_pop_list_train)
			name = ""

			if superpopulations_file:
				scatter_points, colors, markers, edgecolors = \
					plot_coords_by_superpop(coords_by_pop, name, superpopulations_file, plot_legend=False, savefig=False)
				suffix = "_by_superpop"
			else:
				try:
					scatter_points, colors, markers, edgecolors = plot_coords_by_pop(coords_by_pop, name, savefig=False)
					suffix = "_by_pop"
				except:
					scatter_points, colors, markers, edgecolors = plot_coords(encoded_train, name, savefig=False)
					suffix = ""

			scatter_points_per_epoch.append(scatter_points)
			colors_per_epoch.append(colors)
			markers_per_epoch.append(markers)
			edgecolors_per_epoch.append(edgecolors)

		make_animation(epochs, scatter_points_per_epoch, colors_per_epoch,
		               markers_per_epoch, edgecolors_per_epoch,
		               os.path.join(results_directory,
		                            f"dimred_animation{suffix}"))

	if arguments['evaluate']:
		if not isChief:
			print("Work has ended for this worker")
			exit(0)

		chief_print("Evaluating epochs {}".format(epochs))

		# all metrics assumed to have a single value per epoch
		if arguments['metrics']:
			metric_names = arguments['metrics'].split(",")
		else:
			metric_names = ["f1_score_3"]

		metrics = dict()

		for m in metric_names:
			metrics[m] = []

		ind_pop_list_train = read_h5(encoded_data_file, "ind_pop_list_train")
		pop_list = []

		for pop in ind_pop_list_train[:, 1]:
			try:
				pop_list.append(pop.decode("utf-8"))
			except:
				pass

		for epoch in epochs:
			chief_print("########################### epoch {0} ###########################".format(epoch))

			encoded_train = read_h5(encoded_data_file, "{0}_encoded_train".format(epoch))

			coords_by_pop = get_coords_by_pop(data_prefix, encoded_train, ind_pop_list = ind_pop_list_train)

			### count how many f1 scores were doing
			f1_score_order = []
			num_f1_scores = 0
			for m in metric_names:
				if m.startswith("f1_score"):
					num_f1_scores += 1
					f1_score_order.append(m)

			f1_scores_by_pop = {}
			f1_scores_by_pop["order"] = f1_score_order

			for pop in coords_by_pop.keys():
				f1_scores_by_pop[pop] = ["-" for i in range(num_f1_scores)]
			f1_scores_by_pop["avg"] = ["-" for i in range(num_f1_scores)]

			for m in metric_names:

				if m == "hull_error":
					coords_by_pop = get_coords_by_pop(data_prefix, encoded_train, ind_pop_list = ind_pop_list_train)
					n_latent_dim = encoded_train.shape[1]
					if n_latent_dim == 2:
						min_points_required = 3
					else:
						min_points_required = n_latent_dim + 2
					hull_error = convex_hull_error(coords_by_pop, plot=False, min_points_required= min_points_required)
					chief_print("------ hull error : {}".format(hull_error))

					metrics[m].append(hull_error)

				elif m.startswith("f1_score"):
					this_f1_score_index = f1_score_order.index(m)

					k = int(m.split("_")[-1])
					# num_samples_required = np.ceil(k/2.0) + 1 + (k+1) % 2
					num_samples_required = 1

					pops_to_use = get_pops_with_k(num_samples_required, coords_by_pop)

					if len(pops_to_use) > 0 and "{0}_{1}".format(m, pops_to_use[0]) not in metrics.keys():
						for pop in pops_to_use:
							try:
								pop = pop.decode("utf-8")
							except:
								pass
							metric_name_this_pop = "{0}_{1}".format(m, pop)
							metrics[metric_name_this_pop] = []


					f1_score_avg, f1_score_per_pop = f1_score_kNN(encoded_train, pop_list, pops_to_use, k = k)
					chief_print("------ f1 score with {0}NN :{1}".format(k, f1_score_avg))
					metrics[m].append(f1_score_avg)
					assert len(f1_score_per_pop) == len(pops_to_use)
					f1_scores_by_pop["avg"][this_f1_score_index] =  "{:.4f}".format(f1_score_avg)

					for p in range(len(pops_to_use)):
						try:
							pop = pops_to_use[p].decode("utf-8")
						except:
							pop = pops_to_use[p]

						metric_name_this_pop = "{0}_{1}".format(m, pop)
						metrics[metric_name_this_pop].append(f1_score_per_pop[p])
						f1_scores_by_pop[pops_to_use[p]][this_f1_score_index] =  "{:.4f}".format(f1_score_per_pop[p])

				else:
					chief_print("------------------------------------------------------------------------")
					chief_print("Error: Metric {0} is not implemented.".format(m))
					chief_print("------------------------------------------------------------------------")

			write_f1_scores_to_csv(results_directory, "epoch_{0}".format(epoch), superpopulations_file, f1_scores_by_pop, coords_by_pop)

		for m in metric_names:

			plt.plot(epochs, metrics[m], label="train", c="orange")
			plt.xlabel("Epoch")
			plt.ylabel(m)
			plt.savefig(os.path.join(results_directory, m+".pdf"))
			plt.close()

			outfilename = os.path.join(results_directory, m+".csv")
			with open(outfilename, mode='w') as res_file:
				res_writer = csv.writer(res_file, delimiter=',')
				res_writer.writerow(epochs)
				res_writer.writerow(metrics[m])

	if arguments['plot']:
		if not isChief:
			print("Work has ended for this worker")
			exit(0)

		chief_print("Plotting epochs {}".format(epochs))

		ind_pop_list_train = read_h5(encoded_data_file, "ind_pop_list_train")
		pop_list = []

		for pop in ind_pop_list_train[:, 1]:
			try:
				pop_list.append(pop.decode("utf-8"))
			except:
				pass

		for epoch in epochs:
			chief_print("########################### epoch {0} ###########################".format(epoch))

			encoded_train = read_h5(encoded_data_file, "{0}_encoded_train".format(epoch))

			coords_by_pop = get_coords_by_pop(data_prefix, encoded_train, ind_pop_list = ind_pop_list_train)

			if superpopulations_file:

				coords_by_pop = get_coords_by_pop(data_prefix, encoded_train, ind_pop_list = ind_pop_list_train)

				if doing_clustering:
					plot_clusters_by_superpop(coords_by_pop,
					                          os.path.join(results_directory,
					                                       f"clusters_e_{epoch}"),
					                          superpopulations_file,
					                          write_legend = epoch == epochs[0])
				else:
					scatter_points, colors, markers, edgecolors = \
						plot_coords_by_superpop(coords_by_pop,
						                        os.path.join(results_directory,
						                                     f"dimred_e_{epoch}_by_superpop"),
						                        superpopulations_file,
						                        plot_legend = epoch == epochs[0])

			else:
				try:
					plot_coords_by_pop(coords_by_pop,
					                   os.path.join(results_directory,
					                                f"dimred_e_{epoch}_by_pop"))
				except:
					plot_coords(encoded_train,
					            os.path.join(results_directory,
					                         f"dimred_e_{epoch}"))
