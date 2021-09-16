import pytest
import mock

from mlflow.entities import (Experiment, Run, RunInfo, RunData, RunTag, Metric,
                             Param, ExperimentTag, RunStatus, LifecycleStage, ViewType)

experiment = Experiment(experiment_id="1",
                        name="experiment_name",
                        artifact_location="artifact_location",
                        lifecycle_stage=LifecycleStage.ACTIVE,
                        tags=[])
run_info = RunInfo(
    run_uuid="1",
    run_id="1",
    experiment_id="experiment_id",
    user_id="unknown",
    status=RunStatus.to_string(RunStatus.RUNNING),
    start_time=1,
    end_time=None,
    lifecycle_stage=LifecycleStage.ACTIVE,
    artifact_uri="artifact_uri")
run_data = RunData(metrics=[], params=[], tags=[])
run = Run(run_info=run_info, run_data=run_data)

metric = Metric(key="metric1", value=1, timestamp=1, step=1)

param = Param(key="param1", value="val1")

tag = RunTag(key="tag1", value="val1")

experiment_tag = ExperimentTag(key="tag1", value="val1")


@mock.patch("mlflow_elasticsearchstore.elasticsearch_store.ElasticsearchStore.list_experiments")
@pytest.mark.usefixtures('create_mlflow_client')
def test_list_experiments(list_experiments_mock, create_mlflow_client):
    create_mlflow_client.list_experiments(ViewType.ACTIVE_ONLY)
    list_experiments_mock.assert_called_once_with(view_type=ViewType.ACTIVE_ONLY, max_results=None, page_token=None)


@mock.patch("mlflow_elasticsearchstore.elasticsearch_store.ElasticsearchStore.create_experiment")
@pytest.mark.usefixtures('create_mlflow_client')
def test_create_experiment(create_experiment_mock, create_mlflow_client):
    create_experiment_mock.return_value = "1"
    real_id = create_mlflow_client.create_experiment("experiment_name", "artifact_location")
    create_experiment_mock.assert_called_once_with(
        name="experiment_name", artifact_location="artifact_location")
    assert real_id == "1"


@mock.patch("mlflow_elasticsearchstore.elasticsearch_store.ElasticsearchStore.get_experiment")
@pytest.mark.usefixtures('create_mlflow_client')
def test_get_experiment(get_experiment_mock, create_mlflow_client):
    get_experiment_mock.return_value = experiment
    real_experiment = create_mlflow_client.get_experiment("1")
    get_experiment_mock.assert_called_once_with("1")
    assert real_experiment == experiment


@mock.patch("mlflow_elasticsearchstore.elasticsearch_store.ElasticsearchStore.delete_experiment")
@pytest.mark.usefixtures('create_mlflow_client')
def test_delete_experiment(delete_experiment_mock, create_mlflow_client):
    create_mlflow_client.delete_experiment("1")
    delete_experiment_mock.assert_called_once_with("1")


@mock.patch("mlflow_elasticsearchstore.elasticsearch_store.ElasticsearchStore.restore_experiment")
@pytest.mark.usefixtures('create_mlflow_client')
def test_restore_experiment(restore_experiment_mock, create_mlflow_client):
    create_mlflow_client.restore_experiment("1")
    restore_experiment_mock.assert_called_once_with("1")


@mock.patch("mlflow_elasticsearchstore.elasticsearch_store.ElasticsearchStore.rename_experiment")
@pytest.mark.usefixtures('create_mlflow_client')
def test_rename_experiment(rename_experiment_mock, create_mlflow_client):
    create_mlflow_client.rename_experiment("1", "new_name")
    rename_experiment_mock.assert_called_once_with("1", "new_name")


@mock.patch("mlflow_elasticsearchstore.elasticsearch_store.ElasticsearchStore.create_run")
@pytest.mark.usefixtures('create_mlflow_client')
def test_create_run(create_run_mock, create_mlflow_client):
    create_run_mock.return_value = run
    real_run = create_mlflow_client.create_run("experiment_id", 1, [])
    create_run_mock.assert_called_once_with(
        experiment_id="experiment_id", start_time=1, tags=[], user_id='unknown')
    assert real_run == run


@mock.patch("mlflow_elasticsearchstore.elasticsearch_store.ElasticsearchStore.get_run")
@pytest.mark.usefixtures('create_mlflow_client')
def test_get_run(get_run_mock, create_mlflow_client):
    get_run_mock.return_value = run
    real_run = create_mlflow_client.get_run("1")
    get_run_mock.assert_called_once_with("1")
    assert real_run == run


@mock.patch("mlflow_elasticsearchstore.elasticsearch_store.ElasticsearchStore.delete_run")
@pytest.mark.usefixtures('create_mlflow_client')
def test_delete_run(delete_run_mock, create_mlflow_client):
    create_mlflow_client.delete_run("1")
    delete_run_mock.assert_called_once_with("1")


@mock.patch("mlflow_elasticsearchstore.elasticsearch_store.ElasticsearchStore.restore_run")
@pytest.mark.usefixtures('create_mlflow_client')
def test_restore_run(restore_run_mock, create_mlflow_client):
    create_mlflow_client.restore_run("1")
    restore_run_mock.assert_called_once_with("1")


@mock.patch("mlflow_elasticsearchstore.elasticsearch_store.ElasticsearchStore.log_metric")
@pytest.mark.usefixtures('create_mlflow_client')
def test_log_metric(log_metric_mock, create_mlflow_client):
    create_mlflow_client.log_metric("run_id", metric.key, metric.value,
                                    metric.timestamp, metric.step)
    log_metric_mock.assert_called_once_with("run_id", mock.ANY)


@mock.patch("mlflow_elasticsearchstore.elasticsearch_store.ElasticsearchStore.log_param")
@pytest.mark.usefixtures('create_mlflow_client')
def test_log_param(log_param_mock, create_mlflow_client):
    create_mlflow_client.log_param("run_id", param.key, param.value)
    log_param_mock.assert_called_once_with("run_id", mock.ANY)


@mock.patch("mlflow_elasticsearchstore.elasticsearch_store.ElasticsearchStore.set_experiment_tag")
@pytest.mark.usefixtures('create_mlflow_client')
def test_set_experiment_tag(set_experiment_tag_mock, create_mlflow_client):
    create_mlflow_client.set_experiment_tag("run_id", experiment_tag.key, experiment_tag.value)
    set_experiment_tag_mock.assert_called_once_with("run_id", experiment_tag)


@mock.patch("mlflow_elasticsearchstore.elasticsearch_store.ElasticsearchStore.set_tag")
@pytest.mark.usefixtures('create_mlflow_client')
def test_set_tag(set_tag_mock, create_mlflow_client):
    create_mlflow_client.set_tag("run_id", tag.key, tag.value)
    set_tag_mock.assert_called_once_with("run_id", tag)


@mock.patch("mlflow_elasticsearchstore.elasticsearch_store.ElasticsearchStore.get_metric_history")
@pytest.mark.usefixtures('create_mlflow_client')
def test_get_metric_history(get_metric_history_mock, create_mlflow_client):
    create_mlflow_client.get_metric_history("run_id", metric.key)
    get_metric_history_mock.assert_called_once_with(run_id="run_id", metric_key=metric.key)
