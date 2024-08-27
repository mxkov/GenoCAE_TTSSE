import pytest

@pytest.fixture(params=["tiny_01", "tiny_02"])
def f_param_id_test(request):
	return request.param

@pytest.fixture(params=["tiny_01"])
def f_param_id_benchmark(request):
	return request.param
