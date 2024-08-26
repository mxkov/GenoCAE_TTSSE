import glob
import json
import os
import pytest
import shutil
import subprocess
import sys
from utils.data_handler import get_saved_epochs


RESULT_FILES = [
	"losses_from_train.pdf",
	"losses_from_train_t.csv",
	"losses_from_train_v.csv",
	"train_times.csv"
]


def run_gcae_train(**kwargs):
	"""Run run_gcae.py in train mode with given params."""
	cmd = ["python3", "run_gcae.py", "train"]
	for argname in kwargs:
		cmd.append("--" + argname)
		cmd.append(str(kwargs[argname]))

	completed_process = subprocess.run(cmd, text=True)

	if "trainedmodeldir" in kwargs:
		trainedmodeldir = kwargs["trainedmodeldir"]
	else:
		trainedmodeldir = "ae_out"

	# TODO: this is copypasted from run_gcae.py.
	#       make it an importable common func instead.
	trainedmodelname = ".".join(("ae",
	                             kwargs["model_id"],
	                             kwargs["train_opts_id"],
	                             kwargs["data_opts_id"],
		                         kwargs["data"]))
	result_dir = os.path.join(trainedmodeldir, trainedmodelname)

	return completed_process, result_dir


def process_went_well(completed_process):
	"""Check that run_gcae.py finished without errors."""
	try:
		completed_process.check_returncode()
	except Exception as e:
		msg = (f"Process raised {e}\n\n" +
		       f"Full stderr:\n{completed_process.stderr}")
		return False, msg
	return True, ""


def result_exists(result_dir):
	"""Check that result dir exists & contains loss logs & a time log."""
	if not os.path.isdir(result_dir):
		return False, f"Path {result_dir} does not exist"

	missing_files = []
	for file in RESULT_FILES:
		filepath = os.path.join(result_dir, file)
		if not os.path.isfile(filepath):
			missing_files.append(str(filepath))
	if len(missing_files) != 0:
		msg = ("The following output files are missing:" +
		       {", ".join(missing_files)})
		return False, msg

	return True, ""


def check_loss_log(log_file, num_expected_steps):
	"""Check that a loss log contains the expected number of values."""
	f = open(log_file, "r")
	lines = f.read().strip().split("\n")
	f.close()

	if len(lines) != 2:
		return f"Can't parse {log_file}"

	content = {}
	try:
		content["steps"]  = lines[0].strip().split(",")
		content["steps"]  = [float(x) for x in content["steps"]]
		content["losses"] = lines[1].strip().split(",")
		content["losses"] = [float(x) for x in content["losses"]]
	except Exception as e:
		print(f"Can't parse {log_file}", file=sys.stderr)
		raise e

	for x in ("steps", "losses"):
		if len(content[x]) < num_expected_steps:
			msg = (f"File {log_file}: "+
			       f"expected {num_expected_steps} values in {x}, "+
			       f"got {len(content[x])}")
			return msg

	return ""


def result_is_good(result_dir, param_dict):
	"""Check that loss logs are readable & not empty."""
	# TODO: write the logs in column format, write a reader for that,
	#       then rewrite this test accordingly

	epochs   = param_dict["epochs"]
	patience = param_dict["patience"]

	expected_steps = {}
	if patience is None:
		expected_steps["t"] = 1
		expected_steps["v"] = 1
		# these are placeholders.
		# I can't easily calculate the number of tracked steps here for now.
		# so I want to see at least something in the logs.
	else:
		# here training could stop at any moment
		# that we cannot predict in advance,
		# so we want at least something
		expected_steps["t"] = 1
		expected_steps["v"] = 1

	for suffix in ("t", "v"):
		loss_file = os.path.join(result_dir, f"losses_from_train_{suffix}.csv")
		msg = check_loss_log(loss_file, expected_steps[suffix])
		if msg != "":
			return False, msg

	return True, ""


def saved_weights_exist(result_dir, param_dict):
	"""Check that weights dir exists & contains saved model states."""
	# TODO: could also attempt to load each of the saved states.
	#       in another test.
	weights_dir = os.path.join(result_dir, "weights")
	if not os.path.isdir(weights_dir):
		return False, f"Path {weights_dir} does not exist"

	epochs            = param_dict["epochs"]
	resume_from       = param_dict["resume_from"]
	patience          = param_dict["patience"]
	save_interval     = param_dict["save_interval"]
	start_saving_from = param_dict["start_saving_from"]

	def get_effective_epoch(ep):
		return ep + resume_from

	saved_epochs = get_saved_epochs(weights_dir)

	expected_epochs = []
	# 1 model state is always saved after last epoch:
	expected_epochs.append(get_effective_epoch(epochs))
	# + model states that we're instructed to save:
	if patience is None:
		# (if patience is not None, training could stop at any time
		#  and we cannot predict how many of these would be saved)
		expected_epochs.extend([get_effective_epoch(e)
		                        for e in range(1,epochs+1)
		                        if e%save_interval==0 and e>start_saving_from])
	# +1 model state for min valid loss:
	expect_min_valid = (epochs-start_saving_from > 0)
	expected_epochs.sort()

	def fmt_epochs(epoch_list):
		return ", ".join([str(ep) for ep in epoch_list])

	if saved_epochs != expected_epochs:
		msg = (f"Expected saved epochs: [{fmt_epochs(expected_epochs)}], "+
		       f"got saved epochs: [{fmt_epochs(saved_epochs)}]")
		return False, msg
	if expect_min_valid:
		minvalids = glob.glob(os.path.join(weights_dir, "min_valid.*"))
		if len(minvalids) < 2:
			msg = "min_valid.* model weights not found"
			return False, msg

	return True, ""


def parse_params(param_dict):
	"""Convert relevant params to proper type or set them to default value."""
	# epochs is a required param for GCAE-train;
	# if it's invalid or missing, the 1st test should crash
	param_dict["epochs"] = int(param_dict["epochs"])

	defaults = {
		"resume_from"       : False,
		"start_saving_from" : 0,
		"save_interval"     : param_dict["epochs"],
		"patience"          : None
	}
	param_names = list(defaults.keys())
	for param in param_names:
		try:
			param_dict[param] = int(param_dict[param])
		except:
			param_dict[param] = defaults[param]

	return param_dict


def do_cleanup(result_dir):
	"""Remove result dir."""
	shutil.rmtree(result_dir, ignore_errors=True)


def test_run_gcae(benchmark, f_dataset, cleanup=True):

	param_file = os.path.join("test", "test_run_gcae",
	                         f"params_{f_dataset}.json")
	with open(param_file) as pf:
		params = json.load(pf)

	if "cleanup" in params:
		cleanup = params.pop("cleanup")

	# TODO: split all below into separate tests if possible

	compl_proc, result_location = benchmark(run_gcae_train, **params)

	params = parse_params(params)
	# TODO: ideally, I want the same param parser/handler
	#       as the one used in run_gcae.py...
	#       with default param check, type conversion etc.

	passed, else_msg = process_went_well(compl_proc)
	assert passed, else_msg

	passed, else_msg = result_exists(result_location)
	assert passed, else_msg

	passed, else_msg = result_is_good(result_location, params)
	assert passed, else_msg

	passed, else_msg = saved_weights_exist(result_location, params)
	assert passed, else_msg

	if cleanup:
		do_cleanup(result_location)
