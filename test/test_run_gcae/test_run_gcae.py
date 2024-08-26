import json
import os
import pytest
import shutil
import subprocess
import sys


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


def result_is_good(result_dir, param_dict):
	"""Check that loss logs are readable & contain expected number of epochs."""
	# TODO: account for patience!
	# TODO: write the logs in column format, write a reader for that,
	#       then rewrite this test accordingly

	# epochs is a required param for GCAE-train, it HAS to be there
	epochs = int(param_dict["epochs"])

	for suffix in ("t", "v"):
		loss_file = os.path.join(result_dir, f"losses_from_train_{suffix}.csv")
		msg = f"Can't parse f{loss_file}"

		f = open(loss_file, "r")
		lines = f.read().strip().split("\n")
		f.close()

		if len(lines) != 2:
			return False, msg

		content = {}
		try:
			content["steps"]  = lines[0].strip().split(",")
			content["steps"]  = [float(x) for x in content["steps"]]
			content["losses"] = lines[1].strip().split(",")
			content["losses"] = [float(x) for x in content["losses"]]
		except Exception as e:
			print(msg, file=sys.stderr)
			raise e

		for x in ("steps", "losses"):
			if len(content[x]) != epochs:
				msg = (f"File {loss_file}: expected {epochs} values in {x}, "+
				       f"got {len(content[x])}")
				return False, msg

	return True, ""


def saved_weights_exist(result_dir):
	# TODO
	return True, ""


def do_cleanup(result_dir):
	"""Remove result dir."""
	shutil.rmtree(result_dir, ignore_errors=True)


def test_run_gcae(benchmark, f_dataset, cleanup=True):

	param_file = os.path.join("test", "test_run_gcae",
	                         f"params_{f_dataset}.json")
	with open(param_file) as pf:
		params = json.load(pf)

	compl_proc, result_location = benchmark(run_gcae_train, **params)

	passed, else_msg = process_went_well(compl_proc)
	assert passed, else_msg

	passed, else_msg = result_exists(result_location)
	assert passed, else_msg

	passed, else_msg = result_is_good(result_location, params)
	assert passed, else_msg

	if cleanup:
		do_cleanup(result_location)
