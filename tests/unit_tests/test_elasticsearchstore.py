import pytest
import mock
from types import SimpleNamespace
from elasticsearch import Elasticsearch
from elasticsearch.client import IndicesClient
from elasticsearch_dsl import Search, Q

from mlflow.entities import (RunTag, Metric, Param, RunStatus, SourceType,
                             LifecycleStage, ViewType, ExperimentTag)

from mlflow_elasticsearchstore.elasticsearch_store import ElasticsearchStore
from mlflow_elasticsearchstore.models import ExperimentIndex, RunIndex
from mlflow_elasticsearchstore.models import (ElasticExperiment, ElasticRun, ElasticExperimentTag,
                                              ElasticLatestMetric, ElasticMetric,
                                              ElasticParam, ElasticTag)

experiment_response = {'_index': 'mlflow-experiments', '_type': '_doc', '_id': '1',
                       '_version': 1, '_seq_no': 0, '_primary_term': 1, 'found': True,
                       '_source': {'name': 'name',
                                   'lifecycle_stage': 'active',
                                   'artifact_location': 'artifact_location'}}
experiment_deleted_response = {'_index': 'mlflow-experiments', '_type': '_doc', '_id': '1',
                               '_version': 1, '_seq_no': 0, '_primary_term': 1, 'found': True,
                               '_source': {'name': 'name',
                                           'lifecycle_stage': 'deleted',
                                           'artifact_location': 'artifact_location'}}

experiment_tag = ExperimentTag(key="tag1", value="val1")
elastic_experiment_tag = ElasticExperimentTag(key="tag1", value="val1")
experiment = ElasticExperiment(meta={'id': "1"}, name="name",
                               lifecycle_stage=LifecycleStage.ACTIVE,
                               artifact_location="artifact_location")
run_response = {'_index': 'mlflow-runs', '_type': '_doc', '_id': '1',
                '_version': 1, '_seq_no': 0, '_primary_term': 1, 'found': True,
                '_source': {'experiment_id': 'experiment_id', 'user_id': 'user_id',
                            'status': 'RUNNING', 'start_time': 1597324762662,
                            'lifecycle_stage': 'active', 'artifact_uri': 'artifact_location',
                            'latest_metrics': {'metric1': {'value': 1, 'timestamp': 1,
                                                           'step': 1, 'is_nan': False}},
                            'metrics': {'metric1': [{'value': 1, 'timestamp': 1,
                                                     'step': 1, 'is_nan': False}]},
                            'params': {'param1': 'val1'},
                            'tags': {'tag1': 'val1'}}}

deleted_run_response = {'_index': 'mlflow-runs', '_type': '_doc', '_id': '1',
                        '_version': 1, '_seq_no': 0, '_primary_term': 1, 'found': True,
                        '_source': {'experiment_id': 'experiment_id', 'user_id': 'user_id',
                                    'status': 'RUNNING', 'start_time': 1597324762662,
                                    'lifecycle_stage': 'deleted',
                                    'artifact_uri': 'artifact_location',
                                    'latest_metrics': {'metric1': {'value': 1, 'timestamp': 1,
                                                                   'step': 1, 'is_nan': False}},
                                    'metrics': {'metric1': [{'value': 1, 'timestamp': 1,
                                                             'step': 1, 'is_nan': False}]},
                                    'params': {'param1': 'val1'},
                                    'tags': {'tag1': 'val1'}}}
run = ElasticRun(meta={'id': "1"},
                 experiment_id="experiment_id", user_id="user_id",
                 status=RunStatus.to_string(RunStatus.RUNNING),
                 start_time=1, end_time=None,
                 lifecycle_stage=LifecycleStage.ACTIVE, artifact_uri="artifact_location",
                 metrics=[ElasticMetric(key="metric1", value=1,
                                        timestamp=1, step=1, is_nan=False)],
                 latest_metrics=[ElasticLatestMetric(
                     key="metric1", value=1, timestamp=1, step=1, is_nan=False)],
                 params=[ElasticParam(key="param1", value="val1")],
                 tags=[ElasticTag(key="tag1", value="val1")])

deleted_run = ElasticRun(meta={'id': "1"},
                         experiment_id="experiment_id", user_id="user_id",
                         status=RunStatus.to_string(RunStatus.RUNNING),
                         start_time=1, end_time=None,
                         lifecycle_stage=LifecycleStage.DELETED,
                         artifact_uri="artifact_location",
                         metrics=[ElasticMetric(key="metric1", value=1,
                                                timestamp=1, step=1, is_nan=False)],
                         latest_metrics=[ElasticLatestMetric(
                             key="metric1", value=1, timestamp=1, step=1, is_nan=False)],
                         params=[ElasticParam(key="param1", value="val1")],
                         tags=[ElasticTag(key="tag1", value="val1")])

elastic_metric = ElasticMetric(key="metric2", value=2, timestamp=1, step=1, is_nan=False)

metric = Metric(key="metric2", value=2, timestamp=1, step=1)

elastic_param = ElasticParam(key="param2", value="val2")
param = Param(key="param2", value="val2")

elastic_tag = ElasticTag(key="tag2", value="val2")
tag = RunTag(key="tag2", value="val2")


@mock.patch('elasticsearch.Elasticsearch.search')
@pytest.mark.usefixtures('create_store')
def test_list_experiments(search_mock, create_store):
    response = {'hits':
                {'hits': [{'_index': 'mlflow-experiments', '_type': '_doc',
                           '_id': '1', '_score': 0.0,
                           '_source': {'name': 'name',
                                       'lifecycle_stage': 'active',
                                       'artifact_location': 'artifact_location'}}]}}
    search_mock.return_value = response
    real_experiments = create_store.list_experiments()
    body = {
        "query": {
            "bool": {
                "filter": [
                    {"terms": {"lifecycle_stage": ['active']}}
                ]
            }
        }
    }
    search_mock.assert_called_once_with(index=ExperimentIndex.name, body=body)
    mock_experiments = [create_store._dict_to_mlflow_experiment(
        e, e["_id"]) for e in response["hits"]["hits"]]
    assert real_experiments[0].__dict__ == mock_experiments[0].__dict__


@mock.patch('elasticsearch.Elasticsearch.get')
@pytest.mark.usefixtures('create_store')
def test_get_experiment(elastic_get_mock, create_store):
    elastic_get_mock.return_value = experiment_response
    real_experiment = create_store.get_experiment("1")
    elastic_get_mock.assert_called_once_with(index=ExperimentIndex.name, id="1")
    assert create_store._dict_to_mlflow_experiment(
        experiment_response, experiment_response["_id"]).__dict__ == real_experiment.__dict__


@mock.patch('elasticsearch.Elasticsearch.get')
@mock.patch('elasticsearch.Elasticsearch.update')
@pytest.mark.usefixtures('create_store')
def test_delete_experiment(elastic_update_mock, elastic_get_mock, create_store):
    elastic_get_mock.return_value = experiment_response
    create_store.delete_experiment("1")
    elastic_get_mock.assert_called_once_with(index=ExperimentIndex.name, id="1")
    elastic_update_mock.assert_called_once_with(
        index=ExperimentIndex.name, id="1",
        body={"doc": {"lifecycle_stage": LifecycleStage.DELETED}},
        refresh=True)


@mock.patch('elasticsearch.Elasticsearch.get')
@mock.patch('elasticsearch.Elasticsearch.update')
@pytest.mark.usefixtures('create_store')
def test_restore_experiment(elastic_update_mock, elastic_get_mock, create_store):
    elastic_get_mock.return_value = experiment_deleted_response
    create_store.restore_experiment("1")
    elastic_get_mock.assert_called_once_with(index=ExperimentIndex.name, id="1")
    elastic_update_mock.assert_called_once_with(
        index=ExperimentIndex.name, id="1",
        body={"doc": {"lifecycle_stage": LifecycleStage.ACTIVE}},
        refresh=True)


@mock.patch('elasticsearch.Elasticsearch.get')
@mock.patch('elasticsearch.Elasticsearch.update')
@pytest.mark.usefixtures('create_store')
def test_rename_experiment(elastic_update_mock, elastic_get_mock, create_store):
    elastic_get_mock.return_value = experiment_response
    create_store.rename_experiment("1", "new_name")
    elastic_get_mock.assert_called_once_with(index=ExperimentIndex.name, id="1")
    elastic_update_mock.assert_called_once_with(
        index=ExperimentIndex.name, id="1",
        body={"doc": {"name": "new_name"}},
        refresh=True)


@mock.patch('elasticsearch.Elasticsearch.create')
@mock.patch('elasticsearch.Elasticsearch.get')
@mock.patch('uuid.uuid4')
@pytest.mark.usefixtures('create_store')
def test_create_run(uuid_mock, elastic_get_mock,
                    elastic_create_mock, create_store):
    uuid_mock.return_value = SimpleNamespace(hex='run_id')
    elastic_get_mock.return_value = experiment_response
    real_run = create_store.create_run(
        experiment_id="1", user_id="user_id", start_time=1, tags=[])
    uuid_mock.assert_called_once_with()
    elastic_get_mock.assert_called_once_with(index=ExperimentIndex.name, id="1", _source=[
                                             "lifecycle_stage", "artifact_location"])
    body = {"experiment_id": "1",
            "user_id": "user_id",
            "status": RunStatus.to_string(RunStatus.RUNNING),
            "start_time": 1,
            "end_time": None,
            "lifecycle_stage": LifecycleStage.ACTIVE,
            "artifact_uri": 'artifact_location/run_id/artifacts',
            "latest_metrics": {},
            "params": {},
            "tags": {},
            "name": "",
            "source_type": SourceType.to_string(SourceType.UNKNOWN),
            "source_name": "",
            "entry_point_name": "",
            "source_version": "",
            }
    elastic_create_mock.assert_called_once_with(index=RunIndex.name, id="run_id", body=body)
    assert real_run._info.experiment_id == "1"
    assert real_run._info.user_id == "user_id"
    assert real_run._info.start_time == 1
    assert real_run._data.tags == {}
    assert real_run._info.run_id == "run_id"


@mock.patch('elasticsearch.Elasticsearch.update')
@mock.patch('elasticsearch.Elasticsearch.get')
@pytest.mark.usefixtures('create_store')
def test_delete_run(elastic_get_mock, elastic_update_mock, create_store):
    elastic_get_mock.return_value = run_response
    create_store.delete_run("1")
    elastic_get_mock.assert_called_once_with(
        index=RunIndex.name, id="1", _source=["lifecycle_stage"])
    elastic_update_mock.assert_called_once_with(
        index=RunIndex.name, id="1", body={"doc": {"lifecycle_stage": LifecycleStage.DELETED}})


@mock.patch('elasticsearch.Elasticsearch.update')
@mock.patch('elasticsearch.Elasticsearch.get')
@pytest.mark.usefixtures('create_store')
def test_restore_run(elastic_get_mock, elastic_update_mock, create_store):
    elastic_get_mock.return_value = deleted_run_response
    create_store.restore_run("1")
    elastic_get_mock.assert_called_once_with(
        index=RunIndex.name, id="1", _source=["lifecycle_stage"])
    elastic_update_mock.assert_called_once_with(
        index=RunIndex.name, id="1", body={"doc": {"lifecycle_stage": LifecycleStage.ACTIVE}})


@mock.patch('elasticsearch.Elasticsearch.update')
@mock.patch('elasticsearch.Elasticsearch.get')
@pytest.mark.usefixtures('create_store')
def test_update_run_info(elastic_get_mock, elastic_update_mock, create_store):
    elastic_get_mock.return_value = run_response
    create_store.update_run_info("1", RunStatus.FINISHED, 2)
    elastic_get_mock.assert_called_once_with(
        index=RunIndex.name, id="1", _source=["lifecycle_stage"])
    elastic_update_mock.assert_called_once_with(
        index=RunIndex.name, id="1", body={"doc":
                                           {"status": RunStatus.to_string(RunStatus.FINISHED),
                                            "end_time": 2}},
        _source=["experiment_id", "user_id", "start_time", "status",
                 "end_time", "artifact_uri", "lifecycle_stage"])


@mock.patch('elasticsearch.Elasticsearch.get')
@pytest.mark.usefixtures('create_store')
def test_get_run(elastic_get_mock, create_store):
    elastic_get_mock.return_value = run_response
    expected_run = create_store._dict_to_mlflow_run(run_response, run_response["_id"])
    real_run = create_store.get_run("1")
    elastic_get_mock.assert_called_once_with(
        index=RunIndex.name, id="1", _source_excludes=["metrics"])
    assert expected_run._info == real_run._info
    assert expected_run._data._metrics == real_run._data._metrics
    assert expected_run._data._params == real_run._data._params
    assert expected_run._data._tags == real_run._data._tags


@pytest.mark.parametrize("test_metric,test_body",
                         [(Metric(key="metric1", value=2, timestamp=1, step=2,),
                           {"script": {"source": 'ctx._source.latest_metrics.metric1 '
                                       '= params.metric; '
                                       'ctx._source.metrics.metric1.add(params.metric);',
                                       "params": {"metric": {"value": 2,
                                                             "timestamp": 1,
                                                             "step": 2,
                                                             "is_nan": False}}}}),
                          (Metric(key="metric2", value=2, timestamp=1, step=1),
                           {"script": {"source": 'ctx._source.latest_metrics.metric2 '
                                       '= params.metric; '
                                       'ctx._source.metrics.metric2 = [params.metric];',
                                       "params": {"metric": {"value": 2,
                                                             "timestamp": 1,
                                                             "step": 1,
                                                             "is_nan": False}}}})])
@mock.patch('elasticsearch.Elasticsearch.get')
@mock.patch('elasticsearch.Elasticsearch.update')
@pytest.mark.usefixtures('create_store')
def test_log_metric(elastic_update_mock, elastic_get_mock, test_metric, test_body, create_store):
    elastic_get_mock.return_value = run_response
    create_store.log_metric("1", test_metric)
    elastic_get_mock.assert_called_once_with(index=RunIndex.name, id="1", _source=[
                                             "lifecycle_stage", "metrics", "latest_metrics"])
    elastic_update_mock.assert_called_once_with(index=RunIndex.name, id="1", body=test_body)


@mock.patch('elasticsearch.Elasticsearch.get')
@mock.patch('elasticsearch.Elasticsearch.update')
@pytest.mark.usefixtures('create_store')
def test_log_param(elastic_update_mock, elastic_get_mock, create_store):
    elastic_get_mock.return_value = run_response
    create_store.log_param("1", param)
    elastic_get_mock.assert_called_once_with(
        index=RunIndex.name, id="1", _source=["lifecycle_stage"])
    elastic_update_mock.assert_called_once_with(
        index=RunIndex.name, id="1", body={"doc": {"params": {"param2": "val2"}}})


@mock.patch('elasticsearch.Elasticsearch.get')
@mock.patch('elasticsearch.Elasticsearch.update')
@pytest.mark.usefixtures('create_store')
def test_set_experiment_tag(elastic_update_mock, elastic_get_mock, create_store):
    elastic_get_mock.return_value = experiment_response
    create_store.set_experiment_tag("1", experiment_tag)
    elastic_get_mock.assert_called_once_with(
        index=ExperimentIndex.name, id="1", _source=["lifecycle_stage"])
    elastic_update_mock.assert_called_once_with(
        index=ExperimentIndex.name, id="1", body={"doc": {"tags": {"tag1": "val1"}}})


@mock.patch('elasticsearch.Elasticsearch.get')
@mock.patch('elasticsearch.Elasticsearch.update')
@pytest.mark.usefixtures('create_store')
def test_set_tag(elastic_update_mock, elastic_get_mock, create_store):
    elastic_get_mock.return_value = run_response
    create_store.set_tag("1", tag)
    elastic_get_mock.assert_called_once_with(
        index=RunIndex.name, id="1", _source=["lifecycle_stage"])
    elastic_update_mock.assert_called_once_with(
        index=RunIndex.name, id="1", body={"doc": {"tags": {"tag2": "val2"}}})


@pytest.mark.parametrize("test_metric,test_body",
                         [(Metric(key="metric1", value=2, timestamp=1, step=2,),
                           {"script": {"source": 'ctx._source.latest_metrics.metric1 '
                                       '= params.metric; ',
                                       "params": {}}}),
                          (Metric(key="metric1", value=2, timestamp=0, step=0,),
                           {"script": {"source": "", "params": {}}}),
                          (Metric(key="metric2", value=2, timestamp=1, step=1),
                           {"script": {"source": 'ctx._source.latest_metrics.metric2 '
                                       '= params.metric; ',
                                       "params": {}}})])
@pytest.mark.usefixtures('create_store')
def test__update_latest_metric_if_necessary(test_metric, test_body, create_store):
    body = {"script": {"source": "", "params": {}}}
    create_store._update_latest_metric_if_necessary(body, test_metric, run_response)
    assert body == test_body


@pytest.mark.skip(reason="no way of currently testing this")
@pytest.mark.parametrize("test_parsed_filter,test_query,test_type",
                         [({'type': 'parameter', 'key': 'param0',
                             'comparator': 'LIKE', 'value': '%va%'},
                           Q("term", params__key="param0") &
                           Q('bool', must=[Q("wildcard", params__value="*va*")]),
                           "params"),
                          ({'type': 'parameter', 'key': 'param0',
                            'comparator': 'ILIKE', 'value': '%va%'},
                           Q("term", params__key="param0") &
                           Q('bool', must=[Q("wildcard", params__value="*va*")]),
                           "params"),
                          ({'type': 'parameter', 'key': 'param0',
                            'comparator': '=', 'value': 'va'},
                           Q("term", params__key="param0") &
                           Q('bool', must=[Q("term", params__value="va")]),
                           "params"),
                          ({'type': 'metric', 'key': 'metric0', 'comparator': '>', 'value': '1'},
                           Q("term", latest_metrics__key="metric0") &
                           Q('bool', must=[Q("range", latest_metrics__value={'gt': "1"})]),
                           "latest_metrics"),
                          ({'type': 'metric', 'key': 'metric0', 'comparator': '>=', 'value': '1'},
                           Q("term", latest_metrics__key="metric0") &
                           Q('bool', must=[Q("range", latest_metrics__value={'gte': "1"})]),
                           "latest_metrics"),
                          ({'type': 'metric', 'key': 'metric0', 'comparator': '<', 'value': '1'},
                           Q("term", latest_metrics__key="metric0") &
                           Q('bool', must=[Q("range", latest_metrics__value={'lt': "1"})]),
                           "latest_metrics"),
                          ({'type': 'metric', 'key': 'metric0', 'comparator': '<=', 'value': '1'},
                           Q("term", latest_metrics__key="metric0") &
                           Q('bool', must=[Q("range", latest_metrics__value={'lte': "1"})]),
                           "latest_metrics"),
                          ({'type': 'tag', 'key': 'tag0', 'comparator': '!=', 'value': 'val2'},
                           Q("term", tags__key="tag0") &
                           Q('bool', must_not=[Q("term", tags__value="val2")]),
                           "tags")])
@pytest.mark.usefixtures('create_store')
def test___build_elasticsearch_query(test_parsed_filter, test_query,
                                     test_type, create_store):
    actual_query = create_store._build_elasticsearch_query(
        parsed_filters=[test_parsed_filter], s=Search())
    expected_query = Search().query('nested', path=test_type, query=test_query)
    assert actual_query == expected_query


@pytest.mark.skip(reason="no way of currently testing this")
@pytest.mark.usefixtures('create_store')
def test___get_orderby_clauses(create_store):
    order_by_list = ['metrics.`metric0` ASC', 'params.`param0` DESC', 'attributes.start_time ASC']
    actual_query = create_store._get_orderby_clauses(order_by_list=order_by_list, s=Search())
    sort_clauses = [{'latest_metrics.value': {'order': "asc",
                                              "nested": {"path": "latest_metrics",
                                                         "filter": {"term": {'latest_metrics.key':
                                                                             "metric0"}}}}},
                    {'params.value': {'order': "desc",
                                      "nested": {"path": "params",
                                                 "filter": {"term": {'params.key': "param0"}}}}},
                    {"start_time": {'order': "asc"}},
                    {"start_time": {'order': "desc"}},
                    {"_id": {'order': "asc"}}]
    expected_query = Search().sort(*sort_clauses)
    assert actual_query == expected_query
