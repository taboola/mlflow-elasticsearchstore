import pytest

@pytest.mark.usefixtures('create_store')
def test_list_registered_models(create_store):
    models = create_store.list_registered_models(max_results=None, page_token=None)
