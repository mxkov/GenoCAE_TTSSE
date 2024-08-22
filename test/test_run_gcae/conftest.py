import pytest

@pytest.fixture
def f_dataset(request):
	return request.param

def pytest_generate_tests(metafunc):
	metafunc.parametrize("f_dataset", ["tiny", "UKB"])
