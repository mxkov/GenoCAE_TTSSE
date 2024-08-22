import json
import os
import pytest
import subprocess


def run_gcae_train(**kwargs):
	cmd = ["python3", "run_gcae.py", "train"]
	for argname in kwargs:
		cmd.append("--" + argname)
		cmd.append(str(kwargs[argname]))

	completed_process = subprocess.run(cmd, text=True)

	if "trainedmodeldir" in kwargs:
		result_dir = kwargs["trainedmodeldir"]
	else:
		result_dir = "ae_out"
	# TODO: generate subdir name (ae.M1.yadayada)
	return completed_process, result_dir


def process_went_well(completed_process):
	try:
		completed_process.check_returncode()
	except Exception as e:
		assert_msg = (f"Process raised {e}\n\n" +
		              f"Full stderr:\n{completed_process.stderr}")
		return False, assert_msg
	return True, ""


def result_is_good(result_dir):
	if not os.path.isdir(result_dir):
		return False, f"Path {result_dir} does not exist"
	# TODO: check files in result_dir

	return True, ""


def test_run_gcae(benchmark, f_dataset):

	param_file = os.path.join("test", "test_run_gcae",
	                         f"params_{f_dataset}.json")
	with open(param_file) as pf:
		params = json.load(pf)

	compl_proc, result_location = benchmark(run_gcae_train, **params)

	check_flag, else_msg = process_went_well(compl_proc)
	assert check_flag, else_msg

	check_flag, else_msg = result_is_good(result_location)
	assert check_flag, else_msg
