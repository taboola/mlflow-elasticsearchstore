import pytest

@pytest.mark.usefixtures('create_mlflow_client')
def test_list_registered_models( create_mlflow_client):
    models = create_mlflow_client.list_registered_models(max_results=None, page_token=None)