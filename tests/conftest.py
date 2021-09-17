import time
import elasticsearch
import pytest
import mock
from elasticsearch_dsl import connections

from mlflow.tracking import MlflowClient

from mlflow_elasticsearchstore.elasticsearch_store import ElasticsearchStore
from mlflow_elasticsearchstore.models import ElasticExperiment, ElasticRun, ElasticMetric

pytest_plugins = ["docker_compose"]

@pytest.fixture
def create_store():
    connections.create_connection = mock.MagicMock()
    ElasticExperiment.init = mock.MagicMock()
    ElasticRun.init = mock.MagicMock()
    ElasticMetric.init = mock.MagicMock()
    store = ElasticsearchStore("elasticsearch://store_uri", "artifact_uri")
    return store


@pytest.fixture
def create_mlflow_client():
    client = MlflowClient("elasticsearch://tracking_uri", "registry_uri")
    return client


@pytest.fixture
@pytest.mark.usefixtures('start_elastic')
def init_store(start_elastic):
    return ElasticsearchStore(store_uri="elasticsearch://elastic:password@localhost:9200",
                              artifact_uri="viewfs://preprod-pa4/user/mlflow/mlflow_artifacts")


@pytest.fixture(scope="module")
def start_elastic(module_scoped_container_getter, pytestconfig):
    module_scoped_container_getter.get("elastic-search")
    es = elasticsearch.Elasticsearch()
    # wait for yellow status
    for _ in range(100):
        try:
            es.cluster.health(wait_for_status='yellow')
        except Exception:
            time.sleep(1)
    es.snapshot.create_repository(repository="mlflow",  body={"type": "fs", "settings": {"location": "/mount/backups/backup"}})
    es.snapshot.restore("mlflow", "snapshot_1")
    time.sleep(5)
