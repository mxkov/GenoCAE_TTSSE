import os
import pytest


@pytest.fixture
def f_filebase(request):
	return request.param

@pytest.fixture
def f_batch_size(request):
	return request.param

@pytest.fixture
def f_missing_mask(request):
	return request.param

@pytest.fixture
def f_valid_split(request):
	return request.param

@pytest.fixture
def f_impute_missing(request):
	return request.param

@pytest.fixture
def f_sparsifies(request):
	return request.param

@pytest.fixture
def f_norm_opts_flip(request):
	return request.param

@pytest.fixture
def f_norm_opts_missval(request):
	return request.param

@pytest.fixture
def f_pref_chunk_size(request):
	return request.param

@pytest.fixture
def f_shuffle_dataset(request):
	return request.param


def pytest_generate_tests(metafunc):

	# TODO: add more toy examples, incl. incorrect ones
	# TODO: check empty batches (they mess up mask shape)
	fixt_params = {
		"f_filebase"         : [os.path.join("example_tiny", "parquet",
		                                     "HumanOrigins249_tiny")],
		"f_batch_size"       : [20, 13],
		"f_missing_mask"     : [True, False],
		"f_valid_split"      : [0.0, 0.2, 0.9],
		"f_impute_missing"   : [True, False],
		"f_sparsifies"       : [[], [0.0, 0.3]],
		"f_norm_opts_flip"   : [True, False],
		"f_norm_opts_missval": [-1, 9],
		"f_pref_chunk_size"  : [None, 91],
		"f_shuffle_dataset"  : [True, False]
	}
	for fixt_name in fixt_params:
		if fixt_name in metafunc.fixturenames:
				metafunc.parametrize(fixt_name, fixt_params[fixt_name])
