import json
import os

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


def set_tf_config(port="8888"):

	nodelist_str = os.environ["SLURM_JOB_NODELIST"]
	try:
		clust = parse_node_ids(nodelist_str)
	except BadNodelistException:
		raise ValueError(f"Invalid SLURM_JOB_NODELIST value: {nodelist_str}")

	num_workers   = len(clust)
	chief_node_id = clust[0]

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

	return num_workers, chief_node_id
