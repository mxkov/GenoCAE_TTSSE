import pathlib
import sys

TEST_DIR = pathlib.Path(__file__).resolve().parents[0]
sys.path.append(str(TEST_DIR))

from common import (
	load_params,
	run_gcae_train,
	process_went_well,
	do_cleanup
)


def test_run_gcae_benchmark(benchmark, f_dataset):
	"""Benchmark run_gcae.py in train mode."""
	def run_gcae_train_and_cleanup(**param_dict):
		compl_proc, result_location = run_gcae_train(**param_dict)
		success, else_msg = process_went_well(compl_proc)
		assert success, else_msg
		do_cleanup(result_location)

	params = load_params(f_dataset)
	_ = benchmark(run_gcae_train_and_cleanup, **params)
