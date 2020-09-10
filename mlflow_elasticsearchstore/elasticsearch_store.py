import uuid
import math
from typing import List, Tuple, Dict, Any
from elasticsearch_dsl import Search, connections, Q
from elasticsearch import Elasticsearch
from elasticsearch.client import IndicesClient
from elasticsearch.exceptions import NotFoundError
import time
from six.moves import urllib

from mlflow.store.tracking.abstract_store import AbstractStore
from mlflow.store.tracking import SEARCH_MAX_RESULTS_THRESHOLD, SEARCH_MAX_RESULTS_DEFAULT
from mlflow.protos.databricks_pb2 import (INVALID_PARAMETER_VALUE, INVALID_STATE,
                                          INTERNAL_ERROR, RESOURCE_DOES_NOT_EXIST)
from mlflow.entities import (Experiment, RunTag, Metric, Param, Run, RunInfo, RunData,
                             RunStatus, ExperimentTag, LifecycleStage, ViewType, SourceType)
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

from mlflow_elasticsearchstore.models import ExperimentIndex, RunIndex
from mlflow_elasticsearchstore.models import (ElasticExperiment, ElasticRun, ElasticExperimentTag,
                                              ElasticLatestMetric, ElasticMetric,
                                              ElasticParam, ElasticTag)


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
        self.es = Elasticsearch([urllib.parse.urlparse(store_uri).netloc],
                                verify_certs=True)
        self.indices = IndicesClient(self.es)
        ExperimentIndex.init(indices=self.indices)
        RunIndex.init(indices=self.indices)
        super(ElasticsearchStore, self).__init__()

    def _dict_to_mlflow_experiment(self, exp_dict: Dict, _id: str) -> Experiment:
        return Experiment(experiment_id=_id, name=exp_dict["_source"]["name"],
                          artifact_location=exp_dict["_source"]["artifact_location"],
                          lifecycle_stage=exp_dict["_source"]["lifecycle_stage"])

    def _dict_to_mlflow_run(self, run_dict: Dict, _id: str) -> Run:
        return Run(run_info=self._dict_to_mlflow_run_info(run_dict["_source"], _id),
                   run_data=self._dict_to_mlflow_run_data(run_dict["_source"]))

    def _dict_to_mlflow_run_info(self, info_dict: Dict, _id: str) -> RunInfo:
        return RunInfo(run_uuid=_id, run_id=_id,
                       experiment_id=info_dict["experiment_id"],
                       user_id=info_dict["user_id"],
                       status=info_dict["status"],
                       start_time=info_dict["start_time"],
                       end_time=info_dict["end_time"] if "end_time" in info_dict else None,
                       lifecycle_stage=info_dict["lifecycle_stage"],
                       artifact_uri=info_dict["artifact_uri"])

    def _dict_to_mlflow_run_data(self, data_dict: Dict) -> RunData:
        return RunData(metrics=[self._dict_to_mlflow_metric(m_key, m_val) for m_key, m_val in
                                (data_dict["latest_metrics"].items()
                                    if 'latest_metrics' in data_dict else [])],
                       params=[self._dict_to_mlflow_param(p_key, p_val) for p_key, p_val in
                               (data_dict["params"].items() if 'params' in data_dict else [])],
                       tags=[self._dict_to_mlflow_tag(t_key, t_val) for t_key, t_val in
                             (data_dict["tags"].items() if 'tags' in data_dict else[])])

    def _dict_to_mlflow_metric(self, m_key: str, m_val: Dict) -> Metric:
        return Metric(key=m_key, value=m_val["value"], timestamp=m_val["timestamp"],
                      step=m_val["step"])

    def _dict_to_mlflow_param(self, p_key: str, p_val: str) -> Param:
        return Param(key=p_key, value=p_val)

    def _dict_to_mlflow_tag(self, t_key: str, t_val: str) -> RunTag:
        return RunTag(key=t_key, value=t_val)

    def list_experiments(self, view_type: str = ViewType.ACTIVE_ONLY) -> List[Experiment]:
        stages = LifecycleStage.view_type_to_stages(view_type)
        body = {"query": {"bool": {"filter": [{"terms": {"lifecycle_stage": stages}}]}}}
        response = self.es.search(index=ExperimentIndex.name, body=body)
        return [self._dict_to_mlflow_experiment(e, e["_id"]) for e in response["hits"]["hits"]]

    def _list_experiments_name(self) -> List[str]:
        body = {"aggs": {"exp_names": {"terms": {"field": "name"}}}}
        response = self.es.search(index=ExperimentIndex.name,
                                  body=body, size=10000, _source=False)
        return [name["key"] for name in response["aggregations"]["exp_names"]["buckets"]]

    def create_experiment(self, name: str, artifact_location: str = None) -> str:
        if name is None or name == '':
            raise MlflowException('Invalid experiment name', INVALID_PARAMETER_VALUE)
        existing_names = self._list_experiments_name()
        if name in existing_names:
            raise MlflowException('This experiment name already exists', INVALID_PARAMETER_VALUE)
        body = {"name": name,
                "lifecycle_stage": LifecycleStage.ACTIVE,
                "artifact_location": artifact_location
                }
        response = self.es.index(index=ExperimentIndex.name, body=body, refresh=True)
        return str(response["_id"])

    def _check_experiment_is_active(self, experiment: Dict) -> None:
        if experiment["_source"]["lifecycle_stage"] != LifecycleStage.ACTIVE:
            raise MlflowException(
                "The experiment {} must be in the 'active' state. "
                "Current state is {}.".format(
                    experiment["_id"], experiment["_source"]["lifecycle_stage"]),
                INVALID_PARAMETER_VALUE,
            )

    def _get_experiment(self, experiment_id: str, **kwargs: Any) -> Dict:
        try:
            experiment = self.es.get(index=ExperimentIndex.name, id=experiment_id, **kwargs)
        except NotFoundError:
            raise MlflowException(
                "No Experiment with id={} exists".format(experiment_id), RESOURCE_DOES_NOT_EXIST
            )
        return experiment

    def get_experiment(self, experiment_id: str) -> Experiment:
        experiment = self._get_experiment(experiment_id)
        return self._dict_to_mlflow_experiment(experiment, experiment["_id"])

    def delete_experiment(self, experiment_id: str) -> None:
        experiment = self.get_experiment(experiment_id)
        if experiment.lifecycle_stage != LifecycleStage.ACTIVE:
            raise MlflowException('Cannot delete an already deleted experiment.', INVALID_STATE)
        body = {"doc": {"lifecycle_stage": LifecycleStage.DELETED}}
        self.es.update(index=ExperimentIndex.name, id=experiment_id, body=body, refresh=True)

    def restore_experiment(self, experiment_id: str) -> None:
        experiment = self.get_experiment(experiment_id)
        if experiment.lifecycle_stage != LifecycleStage.DELETED:
            raise MlflowException('Cannot restore an active experiment.', INVALID_STATE)
        body = {"doc": {"lifecycle_stage": LifecycleStage.ACTIVE}}
        self.es.update(index=ExperimentIndex.name, id=experiment_id, body=body, refresh=True)

    def rename_experiment(self, experiment_id: str, new_name: str) -> None:
        experiment = self.get_experiment(experiment_id)
        if experiment.lifecycle_stage != LifecycleStage.ACTIVE:
            raise MlflowException('Cannot rename a non-active experiment.', INVALID_STATE)
        body = {"doc": {"name": new_name}}
        self.es.update(index=ExperimentIndex.name, id=experiment_id, body=body, refresh=True)

    def create_run(self, experiment_id: str, user_id: str,
                   start_time: int, tags: List[RunTag]) -> Run:
        run_id = uuid.uuid4().hex
        experiment = self._get_experiment(
            experiment_id, _source=["lifecycle_stage", "artifact_location"])
        self._check_experiment_is_active(experiment)
        artifact_location = append_to_uri_path(experiment["_source"]["artifact_location"], run_id,
                                               ElasticsearchStore.ARTIFACTS_FOLDER_NAME)
        tags_dict = {}
        for tag in tags:
            tags_dict[tag.key] = tag.value
        body = {"experiment_id": experiment_id,
                "user_id": user_id,
                "status": RunStatus.to_string(RunStatus.RUNNING),
                "start_time": start_time,
                "end_time": None,
                "lifecycle_stage": LifecycleStage.ACTIVE,
                "artifact_uri": artifact_location,
                "latest_metrics": {},
                "params": {},
                "tags": tags_dict,
                "name": "",
                "source_type": SourceType.to_string(SourceType.UNKNOWN),
                "source_name": "",
                "entry_point_name": "",
                "source_version": "",
                }
        self.es.create(index=RunIndex.name, id=run_id, body=body)
        run = Run(run_info=self._dict_to_mlflow_run_info(body, run_id),
                  run_data=RunData(metrics=[],
                                   params=[],
                                   tags=tags))
        return run

    def _check_run_is_active(self, run: Dict) -> None:
        if run["_source"]["lifecycle_stage"] != LifecycleStage.ACTIVE:
            raise MlflowException("The run {} must be in the 'active' state. Current state is {}."
                                  .format(run["_id"], run["_source"]["lifecycle_stage"]),
                                  INVALID_PARAMETER_VALUE)

    def _check_run_is_deleted(self, run: Dict) -> None:
        if run["_source"]["lifecycle_stage"] != LifecycleStage.DELETED:
            raise MlflowException("The run {} must be in the 'deleted' state. Current state is {}."
                                  .format(run["_id"], run["_source"]["lifecycle_stage"]),
                                  INVALID_PARAMETER_VALUE)

    def update_run_info(self, run_id: str, run_status: RunStatus, end_time: int) -> RunInfo:
        run = self._get_run(run_id, _source=["lifecycle_stage"])
        self._check_run_is_active(run)
        run.update(status=RunStatus.to_string(run_status), end_time=end_time)
        body = {"doc": {"status": RunStatus.to_string(run_status), "end_time": end_time}}
        info = self.es.update(index=RunIndex.name, id=run_id, body=body,
                              _source=["experiment_id", "user_id", "start_time", "status",
                                       "end_time", "artifact_uri", "lifecycle_stage"])
        return self._dict_to_mlflow_run_info(info["get"]["_source"], info["_id"])

    def get_run(self, run_id: str) -> Run:
        run = self._get_run(run_id, _source_excludes=["metrics"])
        return self._dict_to_mlflow_run(run, run["_id"])

    def _get_run(self, run_id: str, **kwargs: Any) -> Dict:
        try:
            run = self.es.get(index=RunIndex.name, id=run_id, **kwargs)
        except NotFoundError:
            raise MlflowException(
                "Run with id={} not found".format(run_id), RESOURCE_DOES_NOT_EXIST
            )
        return run

    def delete_run(self, run_id: str) -> None:
        run = self._get_run(run_id, _source=["lifecycle_stage"])
        self._check_run_is_active(run)
        body = {"doc": {"lifecycle_stage": LifecycleStage.DELETED}}
        self.es.update(index=RunIndex.name, id=run_id, body=body)

    def restore_run(self, run_id: str) -> None:
        run = self._get_run(run_id, _source=["lifecycle_stage"])
        self._check_run_is_deleted(run)
        body = {"doc": {"lifecycle_stage": LifecycleStage.ACTIVE}}
        self.es.update(index=RunIndex.name, id=run_id, body=body)

    @staticmethod
    def _update_latest_metric_if_necessary(body: Dict, metric: Metric, run: Dict) -> None:
        def _compare_metrics(metric_a: Dict,
                             metric_b: Dict) -> bool:
            return (metric_a["step"], metric_a["timestamp"], metric_a["value"]) > \
                   (metric_b["step"], metric_b["timestamp"], metric_b["value"])
        update_latest_metric = False
        if metric.key in run["_source"]["latest_metrics"]:
            if _compare_metrics(dict(metric), run["_source"]["latest_metrics"][metric.key]):
                update_latest_metric = True
        else:
            update_latest_metric = True
        if update_latest_metric:
            body["script"]["source"] = f'ctx._source.latest_metrics.{metric.key} = params.metric; '

    def _log_metric(self, body: Dict, run: Dict, metric: Metric) -> None:
        _validate_metric(metric.key, metric.value, metric.timestamp, metric.step)
        is_nan = math.isnan(metric.value)
        if is_nan:
            value = 0.
        elif math.isinf(metric.value):
            value = 1.7976931348623157e308 if metric.value > 0 else -1.7976931348623157e308
        else:
            value = metric.value
        self._update_latest_metric_if_necessary(body, metric, run)
        body["script"]["params"]["metric"] = {"value": value,
                                              "timestamp": metric.timestamp,
                                              "step": metric.step,
                                              "is_nan": is_nan}
        if metric.key in run["_source"]["metrics"]:
            body["script"]["source"] += f'ctx._source.metrics.{metric.key}.add(params.metric);'
        else:
            body["script"]["source"] += f'ctx._source.metrics.{metric.key} = [params.metric];'

    def log_metric(self, run_id: str, metric: Metric) -> None:
        run = self._get_run(run_id=run_id, _source=["lifecycle_stage", "metrics", "latest_metrics"])
        self._check_run_is_active(run)
        body: dict = {"script": {"source": "", "params": {}}}
        self._log_metric(body, run, metric)
        self.es.update(index=RunIndex.name, id=run_id, body=body)

    def _log_param(self, body: Dict, param: Param) -> None:
        _validate_param(param.key, param.value)
        body["doc"]["params"][param.key] = param.value

    def log_param(self, run_id: str, param: Param) -> None:
        run = self._get_run(run_id=run_id, _source=["lifecycle_stage"])
        self._check_run_is_active(run)
        body: dict = {"doc": {"params": {}}}
        self._log_param(body, param)
        self.es.update(index=RunIndex.name, id=run_id, body=body)

    def set_experiment_tag(self, experiment_id: str, tag: ExperimentTag) -> None:
        _validate_experiment_tag(tag.key, tag.value)
        experiment = self._get_experiment(experiment_id, _source=["lifecycle_stage"])
        self._check_experiment_is_active(experiment)
        body = {"doc": {"tags": {tag.key: tag.value}}}
        self.es.update(index=ExperimentIndex.name, id=experiment_id, body=body)

    def _set_tag(self, body: Dict, tag: RunTag) -> None:
        _validate_tag(tag.key, tag.value)
        body["doc"]["tags"][tag.key] = tag.value

    def set_tag(self, run_id: str, tag: RunTag) -> None:
        run = self._get_run(run_id=run_id, _source=["lifecycle_stage"])
        self._check_run_is_active(run)
        body: dict = {"doc": {"tags": {}}}
        self._set_tag(body, tag)
        self.es.update(index=RunIndex.name, id=run_id, body=body)

    def get_metric_history(self, run_id: str, metric_key: str) -> List[Metric]:
        history = self._get_run(run_id=run_id, _source=[f'metrics.{metric_key}'])
        return ([self._dict_to_mlflow_metric(metric_key, m_val) for m_val in
                 history["_source"]["metrics"][metric_key]]
                if "metrics" in history["_source"] else [])

    # def list_all_columns(self, experiment_id: str, run_view_type: str) -> 'Columns':
    #     stages = LifecycleStage.view_type_to_stages(run_view_type)
    #     s = Search(index="mlflow-runs").filter("match", experiment_id=experiment_id) \
    #         .filter("terms", lifecycle_stage=stages)
    #     for col in ['latest_metrics', 'params', 'tags']:
    #         s.aggs.bucket(col, 'nested', path=col)\
    #             .bucket(f'{col}_keys', "terms", field=f'{col}.key')
    #     response = s.execute()
    #     metrics = [m.key for m in
    #                response.aggregations.latest_metrics.latest_metrics_keys.buckets]
    #     params = [p.key for p in response.aggregations.params.params_keys.buckets]
    #     tags = [t.key for t in response.aggregations.tags.tags_keys.buckets]
    #     return Columns(metrics=metrics, params=params, tags=tags)

    # def _build_elasticsearch_query(self, parsed_filters: List[dict], s: Search) -> Search:
    #     type_dict = {"metric": "latest_metrics", "parameter": "params", "tag": "tags"}
    #     for search_filter in parsed_filters:
    #         key_type = search_filter.get('type')
    #         key_name = search_filter.get('key')
    #         value = search_filter.get('value')
    #         comparator = search_filter.get('comparator').upper()
    #         filter_ops = {
    #             ">": {'gt': value},
    #             ">=": {'gte': value},
    #             "=": value,
    #             "!=": value,
    #             "<=": {'lte': value},
    #             "<": {'lt': value}
    #         }
    #         if comparator in ["LIKE", "ILIKE"]:
    #             filter_ops[comparator] = f'*{value.split("%")[1]}*'
    #         if key_type == "parameter":
    #             query_type = Q("term", params__key=key_name)
    #             query_val = Q(self.filter_key[comparator][0],
    #                           params__value=filter_ops[comparator])
    #         elif key_type == "tag":
    #             query_type = Q("term", tags__key=key_name)
    #             query_val = Q(self.filter_key[comparator][0], tags__value=filter_ops[comparator])
    #         elif key_type == "metric":
    #             query_type = Q("term", latest_metrics__key=key_name)
    #             query_val = Q(self.filter_key[comparator][0],
    #                           latest_metrics__value=filter_ops[comparator])
    #         if self.filter_key[comparator][1] == "must_not":
    #             query = query_type & Q('bool', must_not=[query_val])
    #         else:
    #             query = query_type & Q('bool', must=[query_val])
    #         s = s.query('nested', path=type_dict[key_type], query=query)
    #     return s

    # def _get_orderby_clauses(self, order_by_list: List[str], s: Search) -> Search:
    #     type_dict = {"metric": "latest_metrics", "parameter": "params", "tag": "tags"}
    #     sort_clauses = []
    #     if order_by_list:
    #         for order_by_clause in order_by_list:
    #             (key_type, key, ascending) = SearchUtils.\
    #                 parse_order_by_for_search_runs(order_by_clause)
    #             sort_order = "asc" if ascending else "desc"
    #             if not SearchUtils.is_attribute(key_type, "="):
    #                 key_type = type_dict[key_type]
    #                 sort_clauses.append({f'{key_type}.value':
    #                                      {'order': sort_order, "nested":
    #                                       {"path": key_type, "filter":
    #                                        {"term": {f'{key_type}.key': key}}}}})
    #             else:
    #                 sort_clauses.append({key: {'order': sort_order}})
    #     sort_clauses.append({"start_time": {'order': "desc"}})
    #     sort_clauses.append({"_id": {'order': "asc"}})
    #     s = s.sort(*sort_clauses)
    #     return s

    # def _search_runs(self, experiment_ids: List[str], filter_string: str,
    #                  run_view_type: str, max_results: int = SEARCH_MAX_RESULTS_DEFAULT,
    #                  order_by: List[str] = None, page_token: str = None,
    #                  columns_to_whitelist: List[str] = None) -> Tuple[List[Run], str]:

    #     def compute_next_token(current_size: int) -> str:
    #         next_token = None
    #         if max_results == current_size:
    #             final_offset = offset + max_results
    #             next_token = SearchUtils.create_page_token(final_offset)
    #         return next_token
    #     if max_results > SEARCH_MAX_RESULTS_THRESHOLD:
    #         raise MlflowException("Invalid value for request parameter max_results. It must be at"
    #                               " most {}, but got value {}"
    #                               .format(SEARCH_MAX_RESULTS_THRESHOLD, max_results),
    #                               INVALID_PARAMETER_VALUE)
    #     stages = LifecycleStage.view_type_to_stages(run_view_type)
    #     parsed_filters = SearchUtils.parse_search_filter(filter_string)
    #     offset = SearchUtils.parse_start_offset_from_page_token(page_token)
    #     s = Search(index="mlflow-runs").filter("match", experiment_id=experiment_ids[0]) \
    #         .filter("terms", lifecycle_stage=stages)
    #     s = self._build_elasticsearch_query(parsed_filters, s)
    #     s = self._get_orderby_clauses(order_by, s)
    #     response = s.source(excludes=["metrics.*"])[offset:offset + max_results].execute()
    #     runs = [self._dict_to_mlflow_run(r) for r in response]
    #     next_page_token = compute_next_token(len(runs))
    #     return runs, next_page_token

    # def log_batch(self, run_id: str, metrics: List[Metric],
    #               params: List[Param], tags: List[RunTag]) -> None:
    #     _validate_run_id(run_id)
    #     _validate_batch_log_data(metrics, params, tags)
    #     _validate_batch_log_limits(metrics, params, tags)
    #     run = self._get_run(run_id=run_id)
    #     self._check_run_is_active(run)
    #     try:
    #         for metric in metrics:
    #             self._log_metric(run, metric)
    #         for param in params:
    #             self._log_param(run, param)
    #         for tag in tags:
    #             self._set_tag(run, tag)
    #         run.save()
    #     except MlflowException as e:
    #         raise e
    #     except Exception as e:
    #         raise MlflowException(e, INTERNAL_ERROR)
