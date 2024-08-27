import json
import os
import shutil
import subprocess


def load_params(param_id):
	"""Read one set of parameters for run_gcae.py from a JSON."""
	param_file = os.path.join("test", "test_run_gcae",
	                         f"params_{param_id}.json")
	with open(param_file) as pf:
		param_dict = json.load(pf)
	return param_dict


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


def do_cleanup(result_dir):
	"""Remove result dir."""
	shutil.rmtree(result_dir, ignore_errors=True)
