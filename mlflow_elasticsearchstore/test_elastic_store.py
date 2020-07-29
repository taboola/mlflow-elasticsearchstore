
from elasticsearch_store import ElasticsearchStore
from models import ElasticExperiment, ElasticRun, ElasticMetric, \
    ElasticParam, ElasticTag, ElasticLatestMetric
import datetime
from mlflow.entities import Metric, Param, RunTag
import time


metric = Metric("metric2", 9, int(datetime.datetime.now().timestamp() * 1000), 0)
param = Param("param2", "val1")
tag = RunTag("tag2", "val1")

store = ElasticsearchStore()
# exp_id = store.create_experiment("a.vivien-store", "artifact_path")
# time.sleep(1)
exp_id = "VRQrmnMBNYJJhinDaGFH"
# run = store.create_run(exp_id, "1", int(datetime.datetime.now().timestamp() * 1000), [])
# run = store._get_run("c92c52c51c934ab8866530e9891dcc7e")
# time.sleep(1)
# store.log_metric(run._info._run_id, metric)
# store.log_param(run._info._run_id, param)
# store.set_tag(run._info._run_id, tag)

# ml_run = store.get_run(run._info._run_id)
# print(ml_run)


runs = store.search_runs([exp_id])

print(runs)
