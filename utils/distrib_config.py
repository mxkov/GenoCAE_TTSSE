import json
import os
import tensorflow as tf
import tensorflow.distribute as tfd
import tensorflow.distribute.experimental as tfde
from tensorflow.python.training.server_lib import ClusterSpec


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


class BadNodelistException(Exception):
	def __init__(self):
		super().__init__()


def parse_node_ids(nodes_str):

	if nodes_str.find("[") == -1:
		return [nodes_str]

	i1 = nodes_str.find("[")+1
	i2 = nodes_str.find("]")
	if i2 == -1:
		raise BadNodelistException()
	node_nums_str = nodes_str[i1:i2]
	prefix        = nodes_str[:i1-1]
	suffix        = nodes_str[i2+1:]

	node_ids = []

	node_num_blocks = node_nums_str.split(",")
	for block in node_num_blocks:
		if block.find("-") == -1:
			new_node_id = prefix + block + suffix
			node_ids.append(new_node_id)
			continue

		node_num_range_ends = block.split("-")
		if len(node_num_range_ends) != 2:
			raise BadNodelistException()
		node_num_range = range(int(node_num_range_ends[0]),
		                       int(node_num_range_ends[1])+1)
		numlen = min([len(x) for x in node_num_range_ends])
		new_node_ids = [prefix + str(n).zfill(numlen) + suffix
		                for n in node_num_range]
		node_ids += new_node_ids

	return node_ids


def node_ids_from_nodelist():
	nodelist_str = os.environ["SLURM_JOB_NODELIST"]
	try:
		cluster = parse_node_ids(nodelist_str)
	except BadNodelistException:
		raise ValueError(f"Invalid SLURM_JOB_NODELIST value: {nodelist_str}")
	return cluster


def get_node_roles():
	clust = node_ids_from_nodelist()
	n_workers   = len(clust)
	chief_node_id = clust[0]
	return n_workers, chief_node_id


def set_tf_config(port="8888"):

	clust = node_ids_from_nodelist()

	current_node_id    = os.environ["SLURMD_NODENAME"]
	current_node_index = clust.index(current_node_id)
	current_node_role  = "worker"

	# TODO: fixed port for all nodes for now
	clust_with_ports = [x+":"+port for x in clust]

	tf_config = {
	  "cluster": {"worker": clust_with_ports},
	  "task"   : {"type": current_node_role, "index": current_node_index}
	}
	os.environ["TF_CONFIG"] = json.dumps(tf_config)

	return tf_config


def set_cluster_env(num_gpus):
	if "CLUSTER" in os.environ:
		cluster_name = os.environ["CLUSTER"]
	else:
		cluster_name = "local"

	if cluster_name == "bianca":
		if num_gpus > 0:
			os.environ["NCCL_DEBUG"] = "INFO"
			os.environ["NCCL_SOCKET_IFNAME"] = "=eth1"
			os.environ["NCCL_P2P_DISABLE"] = "0"
			os.environ["NCCL_P2P_LEVEL"] = "NVL"
	# https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html

	# add your custom setup here


def define_distribution_strategy(multiworker_needed=True):
	# NOTE: assuming homogenous cluster (equal distrib of gpus bw workers).
	#       pretty sure TF makes that assumption as well.

	gpus = tf.config.list_physical_devices(device_type="GPU")
	print("Available GPU devices:\n{}".format(gpus))
	num_physical_gpus = len(gpus)
	gpus = ["gpu:"+ str(i) for i in range(num_physical_gpus)]

	terminate_worker = False

	if "SLURMD_NODENAME" in os.environ:

		n_workers, chief_id = get_node_roles()
		isChief = os.environ["SLURMD_NODENAME"] == chief_id

		set_cluster_env(num_physical_gpus)

		if n_workers > 1 and multiworker_needed:
			#resolver = tfd.cluster_resolver.TFConfigClusterResolver()
			resolver = SlurmClusterResolver_fixed(
			              gpus_per_node=num_physical_gpus)
			if num_physical_gpus > 0:
				comm_impl = tfde.CommunicationImplementation.NCCL
			else:
				comm_impl = tfde.CommunicationImplementation.RING
			#comm_impl = tfde.CommunicationImplementation.AUTO
			comm_opts = tfde.CommunicationOptions(implementation = comm_impl)
			# CollectiveCommunication is deprecated in TF 2.7
			strategy = tfd.MultiWorkerMirroredStrategy(
			                  cluster_resolver = resolver,
			                  communication_options = comm_opts)
		else:
			n_workers = 1
			if not isChief:
				terminate_worker = True
			strategy = tfd.MirroredStrategy(devices = gpus)

	else:
		isChief = True
		n_workers = 1
		strategy = tfd.MirroredStrategy()

	os.environ["isChief"] = json.dumps(int(isChief))

	return strategy, n_workers, terminate_worker
