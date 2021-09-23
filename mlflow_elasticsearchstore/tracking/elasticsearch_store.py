import uuid
import math
from operator import attrgetter
from typing import List, Tuple, Any, Dict
from elasticsearch_dsl import Search, connections, Q
from elasticsearch.exceptions import NotFoundError
from mlflow.store.entities import PagedList
from six.moves import urllib
import ast

from mlflow.store.tracking.abstract_store import AbstractStore
from mlflow.store.tracking import SEARCH_MAX_RESULTS_THRESHOLD, SEARCH_MAX_RESULTS_DEFAULT
from mlflow.protos.databricks_pb2 import (
    INVALID_PARAMETER_VALUE, INVALID_STATE, INTERNAL_ERROR, RESOURCE_DOES_NOT_EXIST)
from mlflow.entities import (Experiment, RunTag, Metric, Param, Run, RunInfo, RunData,
                             RunStatus, ExperimentTag, LifecycleStage, ViewType)
try:
    from mlflow.entities import Columns
except ImportError:
    pass
from mlflow.exceptions import MlflowException
from mlflow.utils.uri import append_to_uri_path
from mlflow.utils.search_utils import SearchUtils
from mlflow.utils.validation import (
    _validate_batch_log_limits,
    _validate_batch_log_data,
    _validate_run_id,
    _validate_metric,
    _validate_param,
    _validate_experiment_tag,
    _validate_tag,
)

from mlflow_elasticsearchstore.tracking.models import (ElasticExperiment, ElasticRun, ElasticMetric,
                                              ElasticParam, ElasticTag,
                                              ElasticLatestMetric, ElasticExperimentTag)


class ElasticsearchStore(AbstractStore):

    ARTIFACTS_FOLDER_NAME = "artifacts"
    DEFAULT_EXPERIMENT_ID = "0"
    filter_key = {
        ">": ["range", "must"],
        ">=": ["range", "must"],
        "=": ["term", "must"],
        "!=": ["term", "must_not"],
        "<=": ["range", "must"],
        "<": ["range", "must"],
        "LIKE": ["wildcard", "must"],
        "ILIKE": ["wildcard", "must"]
    }

    def __init__(self, store_uri: str = None, artifact_uri: str = None) -> None:
        self.is_plugin = True
        self.artifact_root_uri = artifact_uri
        connections.create_connection(hosts=[urllib.parse.urlparse(store_uri).netloc])
        ElasticExperiment.init()
        ElasticRun.init()
        ElasticMetric.init()
        super(ElasticsearchStore, self).__init__()

    def _hit_to_mlflow_experiment(self, hit: Any) -> Experiment:
        return Experiment(experiment_id=hit.meta.id, name=hit.name,
                          artifact_location=hit.artifact_location,
                          lifecycle_stage=hit.lifecycle_stage)

    def _hit_to_mlflow_run(self, hit: Any, columns_to_whitelist_key_dict: dict = None) -> Run:
        return Run(run_info=self._hit_to_mlflow_run_info(hit),
                   run_data=self._hit_to_mlflow_run_data(hit, columns_to_whitelist_key_dict))

    def _hit_to_mlflow_run_info(self, hit: Any) -> RunInfo:
        return RunInfo(run_uuid=hit.run_id, run_id=hit.run_id,
                       experiment_id=str(hit.experiment_id),
                       user_id=hit.user_id,
                       status=hit.status,
                       start_time=hit.start_time,
                       end_time=hit.end_time if hasattr(hit, 'end_time') else None,
                       lifecycle_stage=hit.lifecycle_stage if
                       hasattr(hit, 'lifecycle_stage') else None,
                       artifact_uri=hit.artifact_uri
                       if hasattr(hit, 'artifact_uri') else None)

    def _hit_to_mlflow_run_data(self, hit: Any, columns_to_whitelist_key_dict: dict) -> RunData:
        metrics = [self._hit_to_mlflow_metric(m) for m in
                   (hit.latest_metrics if hasattr(hit, 'latest_metrics') else[])
                   if (columns_to_whitelist_key_dict is None or
                       m.key in columns_to_whitelist_key_dict["metrics"])]
        params = [self._hit_to_mlflow_param(p) for p in
                  (hit.params if hasattr(hit, 'params') else[])
                  if (columns_to_whitelist_key_dict is None or
                      p.key in columns_to_whitelist_key_dict["params"])]
        tags = [self._hit_to_mlflow_tag(t) for t in
                (hit.tags if hasattr(hit, 'tags') else[])
                if (columns_to_whitelist_key_dict is None or
                    t.key in columns_to_whitelist_key_dict["tags"])]
        return RunData(metrics=metrics, params=params, tags=tags)

    def _hit_to_mlflow_metric(self, hit: Any) -> Metric:
        return Metric(key=hit.key,
                      value=hit.value if not (hasattr(hit, 'is_nan')
                                              and hit.is_nan) else float("nan"),
                      timestamp=hit.timestamp, step=hit.step)

    def _hit_to_mlflow_param(self, hit: Any) -> Param:
        return Param(key=hit.key, value=hit.value)

    def _hit_to_mlflow_tag(self, hit: Any) -> RunTag:
        return RunTag(key=hit.key, value=hit.value)

    def list_experiments(self, view_type: str = ViewType.ACTIVE_ONLY, max_results=None, page_token=None) -> List[Experiment]:
        stages = LifecycleStage.view_type_to_stages(view_type)
        response = Search(index="mlflow-experiments").filter("terms",
                                                             lifecycle_stage=stages).execute()
        experiments = [self._hit_to_mlflow_experiment(e) for e in response]
        return PagedList(experiments, None)

    def _list_experiments_name(self) -> List[str]:
        s = Search(index="mlflow-experiments")
        s.aggs.bucket("exp_names", "terms", field="name")
        response = s.params(size=0).execute()
        return [name.key for name in response.aggregations.exp_names.buckets]

    def _get_artifact_location(self, experiment_id: str) -> str:
        return append_to_uri_path(self.artifact_root_uri, str(experiment_id))

    def create_experiment(self, name: str, artifact_location: str = None) -> str:
        if name is None or name == '':
            raise MlflowException('Invalid experiment name', INVALID_PARAMETER_VALUE)
        existing_names = self._list_experiments_name()
        if name in existing_names:
            raise MlflowException('This experiment name already exists', INVALID_PARAMETER_VALUE)

        experiment = ElasticExperiment(name=name, lifecycle_stage=LifecycleStage.ACTIVE,
                                       artifact_location=artifact_location)
        experiment.save(refresh=True)
        if not artifact_location:
            artifact_location = self._get_artifact_location(experiment.meta.id)
        experiment.update(refresh=True, artifact_location=artifact_location)
        return str(experiment.meta.id)

    def _check_experiment_is_active(self, experiment: ElasticExperiment) -> None:
        if experiment.lifecycle_stage != LifecycleStage.ACTIVE:
            raise MlflowException(
                "The experiment {} must be in the 'active' state. "
                "Current state is {}.".format(experiment.meta.id, experiment.lifecycle_stage),
                INVALID_PARAMETER_VALUE,
            )

    def _get_experiment(self, experiment_id: str) -> ElasticExperiment:
        try:
            experiment = ElasticExperiment.get(id=experiment_id)
        except NotFoundError:
            raise MlflowException(
                "No Experiment with id={} exists".format(experiment_id), RESOURCE_DOES_NOT_EXIST
            )
        return experiment

    def get_experiment(self, experiment_id: str) -> Experiment:
        return self._get_experiment(experiment_id).to_mlflow_entity()

    def delete_experiment(self, experiment_id: str) -> None:
        experiment = self._get_experiment(experiment_id)
        if experiment.lifecycle_stage != LifecycleStage.ACTIVE:
            raise MlflowException('Cannot delete an already deleted experiment.', INVALID_STATE)
        experiment.update(refresh=True, lifecycle_stage=LifecycleStage.DELETED)

    def restore_experiment(self, experiment_id: str) -> None:
        experiment = self._get_experiment(experiment_id)
        if experiment.lifecycle_stage != LifecycleStage.DELETED:
            raise MlflowException('Cannot restore an active experiment.', INVALID_STATE)
        experiment.update(refresh=True, lifecycle_stage=LifecycleStage.ACTIVE)

    def rename_experiment(self, experiment_id: str, new_name: str) -> None:
        experiment = self._get_experiment(experiment_id)
        if experiment.lifecycle_stage != LifecycleStage.ACTIVE:
            raise MlflowException('Cannot rename a non-active experiment.', INVALID_STATE)
        experiment.update(refresh=True, name=new_name)

    def create_run(self, experiment_id: str, user_id: str,
                   start_time: int, tags: List[RunTag]) -> Run:
        run_id = uuid.uuid4().hex
        experiment = self._get_experiment(experiment_id)
        self._check_experiment_is_active(experiment)
        artifact_location = append_to_uri_path(experiment.artifact_location, run_id,
                                               ElasticsearchStore.ARTIFACTS_FOLDER_NAME)
        tags_dict = {}
        for tag in tags:
            tags_dict[tag.key] = tag.value
        run_tags = [ElasticTag(key=key, value=value) for key, value in tags_dict.items()]
        run = ElasticRun(meta={'id': run_id},
                         run_id=run_id,
                         experiment_id=experiment_id, user_id=user_id,
                         status=RunStatus.to_string(RunStatus.RUNNING),
                         start_time=start_time, end_time=None,
                         lifecycle_stage=LifecycleStage.ACTIVE, artifact_uri=artifact_location,
                         tags=run_tags)
        run.save()
        return run.to_mlflow_entity()

    def _check_run_is_active(self, run: ElasticRun) -> None:
        if run.lifecycle_stage != LifecycleStage.ACTIVE:
            raise MlflowException("The run {} must be in the 'active' state. Current state is {}."
                                  .format(run.meta.id, run.lifecycle_stage),
                                  INVALID_PARAMETER_VALUE)

    def _check_run_is_deleted(self, run: ElasticRun) -> None:
        if run.lifecycle_stage != LifecycleStage.DELETED:
            raise MlflowException("The run {} must be in the 'deleted' state. Current state is {}."
                                  .format(run.meta.id, run.lifecycle_stage),
                                  INVALID_PARAMETER_VALUE)

    def update_run_info(self, run_id: str, run_status: RunStatus, end_time: int) -> RunInfo:
        run = self._get_run(run_id)
        self._check_run_is_active(run)
        run.update(status=RunStatus.to_string(run_status), end_time=end_time)
        return run.to_mlflow_entity()._info

    def get_run(self, run_id: str) -> Run:
        try:
            run = self._get_run(run_id=run_id)
        except NotFoundError:
            raise MlflowException(
                "Run with id={} not found".format(run_id), RESOURCE_DOES_NOT_EXIST
            )
        return run.to_mlflow_entity()

    def _get_run(self, run_id: str) -> ElasticRun:
        run = ElasticRun.get(id=run_id)
        return run

    def delete_run(self, run_id: str) -> None:
        run = self._get_run(run_id)
        self._check_run_is_active(run)
        run.update(lifecycle_stage=LifecycleStage.DELETED)

    def restore_run(self, run_id: str) -> None:
        run = self._get_run(run_id)
        self._check_run_is_deleted(run)
        run.update(lifecycle_stage=LifecycleStage.ACTIVE)

    @staticmethod
    def _update_latest_metric_if_necessary(new_metric: ElasticMetric, run: ElasticRun) -> None:
        def _compare_metrics(metric_a: ElasticLatestMetric, metric_b: ElasticLatestMetric) -> bool:
            return (metric_a.step, metric_a.timestamp, metric_a.value) > \
                   (metric_b.step, metric_b.timestamp, metric_b.value)
        new_latest_metric = ElasticLatestMetric(key=new_metric.key,
                                                value=new_metric.value,
                                                timestamp=new_metric.timestamp,
                                                step=new_metric.step,
                                                is_nan=new_metric.is_nan)
        latest_metric_exist = False
        for i, latest_metric in enumerate(run.latest_metrics):
            if latest_metric.key == new_metric.key:
                latest_metric_exist = True
                if _compare_metrics(new_latest_metric, latest_metric):
                    run.latest_metrics[i] = new_latest_metric
        if not (latest_metric_exist):
            run.latest_metrics.append(new_latest_metric)

    def _log_metric(self, run: ElasticRun, metric: Metric) -> None:
        _validate_metric(metric.key, metric.value, metric.timestamp, metric.step)
        is_nan = math.isnan(metric.value)
        if is_nan:
            value = 0.
        elif math.isinf(metric.value):
            value = 1.7976931348623157e308 if metric.value > 0 else -1.7976931348623157e308
        else:
            value = metric.value
        new_metric = ElasticMetric(key=metric.key,
                                   value=value,
                                   timestamp=metric.timestamp,
                                   step=metric.step,
                                   is_nan=is_nan,
                                   run_id=run.run_id)
        self._update_latest_metric_if_necessary(new_metric, run)
        new_metric.save()

    def log_metric(self, run_id: str, metric: Metric) -> None:
        run = self._get_run(run_id=run_id)
        self._check_run_is_active(run)
        self._log_metric(run, metric)
        run.update(latest_metrics=run.latest_metrics)

    def _log_param(self, run: ElasticRun, param: Param) -> None:
        _validate_param(param.key, param.value)
        new_param = ElasticParam(key=param.key,
                                 value=param.value)
        run.params.append(new_param)

    def log_param(self, run_id: str, param: Param) -> None:
        run = self._get_run(run_id=run_id)
        self._check_run_is_active(run)
        self._log_param(run, param)
        run.update(params=run.params)

    def set_experiment_tag(self, experiment_id: str, tag: ExperimentTag) -> None:
        _validate_experiment_tag(tag.key, tag.value)
        experiment = self._get_experiment(experiment_id)
        self._check_experiment_is_active(experiment)
        new_tag = ElasticExperimentTag(key=tag.key, value=tag.value)
        experiment.tags.append(new_tag)
        experiment.update(tags=experiment.tags)

    def _set_tag(self, run: ElasticRun, tag: RunTag) -> None:
        _validate_tag(tag.key, tag.value)
        new_tag = ElasticTag(key=tag.key,
                             value=tag.value)
        run.tags.append(new_tag)

    def set_tag(self, run_id: str, tag: RunTag) -> None:
        run = self._get_run(run_id=run_id)
        self._check_run_is_active(run)
        self._set_tag(run, tag)
        run.update(tags=run.tags)

    def get_metric_history(self, run_id: str, metric_key: str) -> List[Metric]:
        s = Search(index="mlflow-metrics").filter("term", run_id=run_id) \
            .filter("term", key=metric_key)
        return [self._hit_to_mlflow_metric(m) for m in s.scan()]

    def _list_columns(self, experiment_id: str, stages: List[LifecycleStage],
                      column_type: str, columns: List[str], size: int = 100) -> None:
        s = Search(index="mlflow-runs").filter("match", experiment_id=experiment_id) \
            .filter("terms", lifecycle_stage=stages)
        s.aggs.bucket(column_type, 'nested', path=column_type) \
            .bucket(f'{column_type}_keys', "composite", size=size,
                    sources=[{"key": {"terms": {"field": f'{column_type}.key'}}}])
        response = s.params(size=0).execute()
        new_columns = [column.key.key for column in attrgetter(
            f'aggregations.{column_type}.{column_type}_keys.buckets')(response)]
        columns += new_columns
        while (len(new_columns) == size):
            last_col = attrgetter(
                f'aggregations.{column_type}.{column_type}_keys.after_key.key')(response)
            s = Search(index="mlflow-runs").filter("match", experiment_id=experiment_id) \
                .filter("terms", lifecycle_stage=stages)
            s.aggs.bucket(column_type, 'nested', path=column_type) \
                .bucket(f'{column_type}_keys', "composite", size=size,
                        sources=[{"key": {"terms": {"field": f'{column_type}.key'}}}],
                        after={"key": last_col})
            response = s.params(size=0).execute()
            new_columns = [column.key.key for column in attrgetter(
                f'aggregations.{column_type}.{column_type}_keys.buckets')(response)]
            columns += new_columns

    def list_all_columns(self, experiment_id: str, run_view_type: str) -> 'Columns':
        columns: Dict[str, List[str]] = {"latest_metrics": [],
                                         "params": [],
                                         "tags": []}
        stages = LifecycleStage.view_type_to_stages(run_view_type)
        for column_type in ['latest_metrics', 'params', 'tags']:
            self._list_columns(experiment_id, stages, column_type,
                               columns[column_type])
        return Columns(metrics=columns['latest_metrics'],
                       params=columns['params'],
                       tags=columns['tags'])

    def _build_columns_to_whitelist_key_dict(self, columns_to_whitelist: List[str]) -> dict:
        if columns_to_whitelist is None:
            return None
        columns_to_whitelist_key_dict: dict = {"metrics": set(), "params": set(), "tags": set()}
        for col in columns_to_whitelist:
            word = col.split(".")
            key = ".".join(word[1:])
            columns_to_whitelist_key_dict[word[0]].add(key)
        return columns_to_whitelist_key_dict

    def _build_elasticsearch_query(self, parsed_filters: List[dict]) -> List[Q]:
        type_dict = {"metric": "latest_metrics", "parameter": "params", "tag": "tags"}
        search_query = []
        for search_filter in parsed_filters:
            key_type = search_filter.get('type')
            key_name = search_filter.get('key')
            value = search_filter.get('value')
            comparator = search_filter.get('comparator').upper()
            filter_ops = {
                ">": {'gt': value},
                ">=": {'gte': value},
                "=": value,
                "!=": value,
                "<=": {'lte': value},
                "<": {'lt': value}
            }
            if comparator in ["LIKE", "ILIKE"]:
                filter_ops[comparator] = f'*{value.split("%")[1]}*'
            if key_type == "parameter":
                query_type = Q("term", params__key=key_name)
                query_val = Q(self.filter_key[comparator][0], params__value=filter_ops[comparator])
            elif key_type == "tag":
                query_type = Q("term", tags__key=key_name)
                query_val = Q(self.filter_key[comparator][0], tags__value=filter_ops[comparator])
            elif key_type == "metric":
                query_type = Q("term", latest_metrics__key=key_name)
                query_val = Q(self.filter_key[comparator][0],
                              latest_metrics__value=filter_ops[comparator])
            if self.filter_key[comparator][1] == "must_not":
                query = Q('bool', filter=[query_type], must_not=[query_val])
            else:
                query = Q('bool', filter=[query_type, query_val])
            search_query.append(Q('nested', path=type_dict[key_type], query=query))
        return search_query

    def _get_orderby_clauses(self, order_by_list: List[str]) -> List[dict]:
        type_dict = {"metric": "latest_metrics", "parameter": "params", "tag": "tags"}
        sort_clauses = []
        if order_by_list:
            for order_by_clause in order_by_list:
                (key_type, key, ascending) = SearchUtils. \
                    parse_order_by_for_search_runs(order_by_clause)
                sort_order = "asc" if ascending else "desc"
                if not SearchUtils.is_attribute(key_type, "="):
                    key_type = type_dict[key_type]
                    sort_clauses.append({f'{key_type}.value':
                                         {'order': sort_order, "nested":
                                          {"path": key_type, "filter":
                                           {"term": {f'{key_type}.key': key}}}}})
                else:
                    sort_clauses.append({key: {'order': sort_order}})
        sort_clauses.append({"start_time": {'order': "desc"}})
        sort_clauses.append({"run_id": {'order': "asc"}})
        return sort_clauses

    def _search_runs(self, experiment_ids: List[str], filter_string: str,
                     run_view_type: str, max_results: int = SEARCH_MAX_RESULTS_DEFAULT,
                     order_by: List[str] = None, page_token: str = None,
                     columns_to_whitelist: List[str] = None) -> Tuple[List[Run], str]:

        if max_results > 10000:
            raise MlflowException("Invalid value for request parameter max_results. It must be at "
                                  "most {}, but got value {}"
                                  .format(10000, max_results),
                                  INVALID_PARAMETER_VALUE)
        stages = LifecycleStage.view_type_to_stages(run_view_type)
        parsed_filters = SearchUtils.parse_search_filter(filter_string)
        filter_queries = [Q("match", experiment_id=experiment_ids[0]),
                          Q("terms", lifecycle_stage=stages)]
        filter_queries += self._build_elasticsearch_query(parsed_filters)
        sort_clauses = self._get_orderby_clauses(order_by)
        s = Search(index="mlflow-runs").query('bool', filter=filter_queries)
        s = s.sort(*sort_clauses)
        if page_token != "" and page_token is not None:
            s = s.extra(search_after=ast.literal_eval(page_token))
        response = s.params(size=max_results).execute()
        columns_to_whitelist_key_dict = self._build_columns_to_whitelist_key_dict(
            columns_to_whitelist)
        runs = [self._hit_to_mlflow_run(hit, columns_to_whitelist_key_dict) for hit in response]
        if len(runs) == max_results:
            next_page_token = response.hits.hits[-1].sort
        else:
            next_page_token = []
        return runs, str(next_page_token)

    def update_artifacts_location(self, run_id: str, new_artifacts_location: str) -> None:
        run = self._get_run(run_id=run_id)
        run.update(artifact_uri=new_artifacts_location)

    def log_batch(self, run_id: str, metrics: List[Metric],
                  params: List[Param], tags: List[RunTag]) -> None:
        _validate_run_id(run_id)
        _validate_batch_log_data(metrics, params, tags)
        _validate_batch_log_limits(metrics, params, tags)
        run = self._get_run(run_id=run_id)
        self._check_run_is_active(run)
        try:
            for metric in metrics:
                self._log_metric(run, metric)
            for param in params:
                self._log_param(run, param)
            for tag in tags:
                self._set_tag(run, tag)
            run.save()
        except MlflowException as e:
            raise e
        except Exception as e:
            raise MlflowException(e, INTERNAL_ERROR)
