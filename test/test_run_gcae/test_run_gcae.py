import glob
import os
import pathlib
import sys

script_path = pathlib.Path(__file__).resolve()
GCAE_DIR = script_path.parents[2]
TEST_DIR = script_path.parents[0]
for dirpath in (GCAE_DIR, TEST_DIR):
	sys.path.append(str(dirpath))

from utils.data_handler import get_saved_epochs
from common import (
	load_params,
	run_gcae_train,
	process_went_well,
	do_cleanup
)


RESULT_FILES = [
	"losses_from_train.pdf",
	"losses_from_train_t.csv",
	"losses_from_train_v.csv",
	"train_times.csv"
]


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
		new_epochs = [get_effective_epoch(e)
		              for e in range(1,epochs+1)
		              if e%save_interval==0 and e>start_saving_from]
		new_epochs = [ep for ep in new_epochs if ep not in expected_epochs]
		expected_epochs.extend(new_epochs)
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


def test_run_gcae(f_dataset, cleanup=True):
	"""Run run_gcae.py in train mode, make sure it runs, check the output."""
	params = load_params(f_dataset)
	if "cleanup" in params:
		cleanup = params.pop("cleanup")

	compl_proc, result_location = run_gcae_train(**params)

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
