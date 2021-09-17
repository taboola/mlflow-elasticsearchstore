import time
import elasticsearch
import pytest

pytest_plugins = ["docker_compose"]


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
