import time
import logging
import ast
from versioneer import get_config_from_root

from elasticsearch_dsl import Search, connections, Q
from elasticsearch.exceptions import NotFoundError

from six.moves import urllib
from typing import List, Tuple, Any, Dict

from mlflow.entities.model_registry.model_version_stages import (
    STAGE_NONE,
    get_canonical_stage,
    DEFAULT_STAGES_FOR_GET_LATEST_VERSIONS,
    STAGE_DELETED_INTERNAL,
    STAGE_ARCHIVED,
)

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INVALID_PARAMETER_VALUE,
    RESOURCE_ALREADY_EXISTS,
    INVALID_STATE,
    RESOURCE_DOES_NOT_EXIST,
)
import mlflow.store.db.utils
from mlflow.store.model_registry import (
    SEARCH_REGISTERED_MODEL_MAX_RESULTS_DEFAULT,
    SEARCH_REGISTERED_MODEL_MAX_RESULTS_THRESHOLD,
)
from mlflow.store.db.base_sql_model import Base
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.model_registry.abstract_store import AbstractStore
from mlflow_elasticsearchstore.model_registry.models import (ElasticRegisteredModel, ElasticRegisteredModelTag,
                                              ElasticModelVersion, ElasticModelVersionTag)
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus

from mlflow.utils.search_utils import SearchUtils
from mlflow.utils.uri import extract_db_type_from_uri
from mlflow.utils.validation import (
    _validate_registered_model_tag,
    _validate_model_version_tag,
    _validate_model_name,
    _validate_model_version,
    _validate_tag_name,
)

from mlflow.entities.model_registry import (
    RegisteredModel,
    ModelVersion,
    RegisteredModelTag,
    ModelVersionTag,
)

_logger = logging.getLogger(__name__)

def now():
    return int(time.time() * 1000)

class ElasticsearchStore(AbstractStore):

    CREATE_MODEL_VERSION_RETRIES = 3

    def __init__(self, store_uri):
        self.is_plugin = True
        connections.create_connection(hosts=[urllib.parse.urlparse(store_uri).netloc])
        ElasticRegisteredModel.init()
        super(ElasticsearchStore, self).__init__()

    def _get_registered_model(self, name) -> ElasticRegisteredModel:
        try:
            registered_model = ElasticRegisteredModel.get(id=name)
        except NotFoundError:
            raise MlflowException(
                "No Experiment with name={} exists".format(name), RESOURCE_DOES_NOT_EXIST
            )
        return registered_model
    
    def _registered_model_exists(self, name) -> bool:
        try:
            ElasticRegisteredModel.get(id=name)
            return True
        except NotFoundError:
            return False

    def create_registered_model(self, name, tags=None, description=None) -> RegisteredModel:
        """
        Create a new registered model in backend store.
        :param name: Name of the new model. This is expected to be unique in the backend store.
        :param tags: A list of :py:class:`mlflow.entities.model_registry.RegisteredModelTag`
                     instances associated with this registered model.
        :param description: Description of the version.
        :return: A single object of :py:class:`mlflow.entities.model_registry.RegisteredModel`
                 created in the backend.
        """
        creation_time = now()

        _validate_model_name(name)
        for tag in tags or []:
            _validate_registered_model_tag(tag.key, tag.value)
        
        if self._registered_model_exists(name):
            raise MlflowException(
                    "Registered Model (name={}) already exists. ".format(name),
                    RESOURCE_ALREADY_EXISTS,
            )

        tags_dict = {}
        for tag in tags:
            tags_dict[tag.key] = tag.value

        registered_model_tags = [ElasticRegisteredModelTag(key=key, value=value) for key, value in tags_dict.items()]
        registered_model = ElasticRegisteredModel(
            meta={'id': name},
            name=name,
            creation_time=creation_time,
            last_updated_time=creation_time,
            description=description,
            registered_model_tags=registered_model_tags
        )
        registered_model.save(refresh=True)
        return registered_model.to_mlflow_entity()
    
    def update_registered_model(self, name, description) -> RegisteredModel:
        updated_time = now()
        registered_model = self._get_registered_model(name)
        registered_model.update(refresh=True, description=description, last_updated_time=updated_time)
        return registered_model.to_mlflow_entity()
    
    def rename_registered_model(self, name, new_name) -> RegisteredModel:
        updated_time = now()
        registered_model = self._get_registered_model(name)
        registered_model.update(refresh=True, _id=new_name, last_updated_time=updated_time)
        return registered_model.to_mlflow_entity()

    def delete_registered_model(self, name) -> None:
        registered_model = self._get_registered_model(name)
        registered_model.delete()

    def list_registered_models(self, max_results, page_token) -> List[RegisteredModel]:
        return self.search_registered_models(max_results=max_results, page_token=page_token)
    
    def _parse_filter_string(self, filter_string:str):
        splits = filter_string.split(' ', 2)
        return splits[0], splits[1], splits[2]
    
    def _parse_order_by_string(self, order_by):
        splits = order_by.split(' ', 1)
        order_by_str = splits[0]
        if order_by_str == 'timestamp':
            order_by_str = 'last_updated_time'
        if len(splits) > 1 and splits[1] == 'DESC':
            return "-" + order_by_str
        else:
            return order_by_str
    
    def _search_registered_models(self, filter_string, max_results, order_by, page_token):
        if max_results > 10000:
            raise MlflowException("Invalid value for request parameter max_results. It must be at "
                                  "most {}, but got value {}"
                                  .format(10000, max_results),
                                  INVALID_PARAMETER_VALUE)
        _,_, value = self._parse_filter_string(filter_string=filter_string)
        s = Search(doc_type=[ElasticRegisteredModel], index="mlflow-registered-model")
        print(value.replace("'",""))
        s = s.query("wildcard", name=value.replace("'","").replace("%","*"))
        print(order_by)
        s = s.sort(self._parse_order_by_string(order_by=order_by[0]))
        if page_token != "" and page_token is not None:
            s = s.extra(search_after=ast.literal_eval(page_token))
        
        response = s.params(size=max_results).execute()

        registered_models = [elastic_registered_model.to_mlflow_entity() for elastic_registered_model in response]
        if len(registered_models) == max_results:
            next_page_token = response.hits.hits[-1].sort
        else:
            next_page_token = []
        return registered_models, str(next_page_token)
    

    def search_registered_models(self, filter_string=None, max_results=None, order_by=None, page_token=None):
        models, token = self._search_registered_models(filter_string=filter_string, max_results=max_results, order_by=order_by, page_token=page_token)
        return PagedList(models, token)

    def get_registered_model(self, name) -> RegisteredModel:
        return self._get_registered_model(name).to_mlflow_entity()

    def get_latest_versions(self, name, stages=None) -> List[ModelVersion]:
        registered_model = self.get_registered_model(name)
        latest_versions = registered_model.latest_versions
        if stages is None or len(stages) == 0:
            expected_stages = set(
                [get_canonical_stage(stage) for stage in DEFAULT_STAGES_FOR_GET_LATEST_VERSIONS]
            )
        else:
            expected_stages = set([get_canonical_stage(stage) for stage in stages])
        return [mv for mv in latest_versions if mv.current_stage in expected_stages]

    def set_registered_model_tag(self, name, tag):
        _validate_model_name(name)
        _validate_registered_model_tag(tag.key, tag.value)
        
        registered_model = self._get_registered_model(name)

        tags = registered_model.registered_model_tags
        tags_dict = {}
        for t in tags or []:
            tags_dict[t.key] = t.value

        tags_dict[tag.key] = tag.value
        registered_model_tags = [ElasticRegisteredModelTag(key=key, value=value) for key, value in tags_dict.items()]
        registered_model.update(refresh=True, registered_model_tags=registered_model_tags)

    def delete_registered_model_tag(self, name, key):
        _validate_model_name(name)

        registered_model = self._get_registered_model(name)
        tags = registered_model.registered_model_tags
        tags_dict = {}
        for tag in tags or []:
            tags_dict[tag.key] = tag.value

        # Delete tag if it exists
        tags_dict.pop(key, None)
        
        registered_model_tags = [ElasticRegisteredModelTag(key=key, value=value) for key, value in tags_dict.items()]
        registered_model.update(refresh=True, registered_model_tags=registered_model_tags)

    def create_model_version(self, name, source, run_id=None, tags=None, run_link=None, description=None):
        # Function to derive next version
        def next_version(elastic_registered_model):
            if elastic_registered_model.model_versions:
                return max([mv.version for mv in elastic_registered_model.model_versions]) + 1
            else:
                return 1

        _validate_model_name(name)

        tags_dict = {}
        for tag in tags or []:
            _validate_model_version_tag(tag.key, tag.value)
            tags_dict[tag.key] = tag.value
        
        creation_time = now()
        elastic_registered_model = self._get_registered_model(name)
        version = next_version(elastic_registered_model)

        model_version = ElasticModelVersion(
            name=name,
            version=version,
            creation_time=creation_time,
            last_updated_time=creation_time,
            source=source,
            run_id=run_id,
            run_link=run_link,
            description=description,
            current_stage=STAGE_NONE,
            status=ModelVersionStatus.to_string(ModelVersionStatus.READY),
            model_version_tags=[
                        ElasticModelVersionTag(key=key, value=value) for key, value in tags_dict.items()
                    ]
        )

        model_versions = elastic_registered_model.model_versions or []
        model_versions.append(model_version)
        elastic_registered_model.update(refresh=True, model_versions=model_versions, last_updated_time=creation_time)
        return model_version.to_mlflow_entity()

    def _get_all_model_versions(self, name):
        elastic_registered_model = self._get_registered_model(name)
        elastic_model_versions = elastic_registered_model.model_versions
        return elastic_registered_model, elastic_model_versions

    def _get_elastic_model_version(self, name, version):
        # TODO make this a more efficient elastic-based query
        ver = int(version)
        elastic_registered_model = self._get_registered_model(name=name)
        for v in elastic_registered_model.model_versions:
            if v.version == ver and v.current_stage != STAGE_DELETED_INTERNAL:
                return (elastic_registered_model, v)
        return (None, None)   

    def update_model_version(self, name, version, description):
        updated_time = now()
        _validate_model_name(name)
        _validate_model_version(version)
        elastic_registered_model = self._get_registered_model(name)
        
        for elastic_model_version in elastic_registered_model.model_versions or []:
            if elastic_model_version.version == int(version):        
                elastic_model_version.description = description
                elastic_model_version.last_updated_time = updated_time
        
        elastic_registered_model.update(refresh=True, model_versions=elastic_registered_model.model_versions)
        return elastic_model_version.to_mlflow_entity()
    
    def transition_model_version_stage(self, name, version, stage, archive_existing_versions):
        is_active_stage = get_canonical_stage(stage) in DEFAULT_STAGES_FOR_GET_LATEST_VERSIONS
        if archive_existing_versions and not is_active_stage:
            msg_tpl = (
                "Model version transition cannot archive existing model versions "
                "because '{}' is not an Active stage. Valid stages are {}"
            )
            raise MlflowException(msg_tpl.format(stage, DEFAULT_STAGES_FOR_GET_LATEST_VERSIONS))
        
        last_updated_time = now()
        _validate_model_name(name)
        _validate_model_version(version)
        
        elastic_registered_model, elastic_model_versions = self._get_all_model_versions(name)

        elastic_model_version = None
        for mv in elastic_model_versions:
            if archive_existing_versions:
                if mv.version != int(version) and mv.current_stage == get_canonical_stage(stage):
                    mv.current_stage = STAGE_ARCHIVED
                    mv.last_updated_time = last_updated_time
            if mv.version == int(version):
                mv.current_stage = get_canonical_stage(stage)
                mv.last_updated_time = last_updated_time
                elastic_model_version = mv
        
        elastic_registered_model.update(refresh=True, last_updated_time=last_updated_time, model_versions=elastic_model_versions)
        return elastic_model_version.to_mlflow_entity()

    def delete_model_version(self, name, version) -> None:
        last_updated_time = now()
        _validate_model_name(name)
        _validate_model_name(version)
        elastic_registered_model = self._get_registered_model(name)
        model_versions = elastic_registered_model.model_versions
        for mv in model_versions:
            if mv.version == int(version):
                mv.current_stage = STAGE_DELETED_INTERNAL
                mv.last_updated_time = last_updated_time
                mv.description = None
                mv.user_id = None
                mv.source = "REDACTED-SOURCE-PATH"
                mv.run_id = "REDACTED-RUN-ID"
                mv.run_link = "REDACTED-RUN-LINK"
                mv.status_message = None
        elastic_registered_model.update(refresh=True, last_updated_time=last_updated_time, model_versions=model_versions)

    def get_model_version(self, name, version):
        _, version = self._get_elastic_model_version(name, version)
        if version is None:
            return None
        else:
            return version.to_mlflow_entity()

    def get_model_version_download_uri(self, name, version):
        _,version = self._get_elastic_model_version(name, version)
        return version.source

    def _process_inner_hits(self, hit):
        versions = []
        for v in hit.meta.inner_hits.model_versions:
            print(v.version)
            versions.append(
                ModelVersion(
                            name=v.name,
                            version=v.version,
                            creation_timestamp=v.creation_time,
                            last_updated_timestamp=v.last_updated_time,
                            description=v.description,
                            user_id= v.user_id if hasattr(v, "user_id") else None,
                            current_stage=v.current_stage,
                            source=v.source,
                            run_id=v.run_id,
                            status=v.status if hasattr(v, "status") else None,
                            status_message=v.status_message if hasattr(v, "status_message") else None,
                            tags=[ModelVersionTag(key=t.key,value=t.value) for t in v.model_version_tags or []] if hasattr(v,"model_version_tags") else [],
                            run_link=v.run_link,
                )
            )
        return versions


    
    def _parse_model_version_string(self, filter_string:str):
        if "IN" in filter_string:
            splits = filter_string.split(" IN ", 1)
            str_list = splits[1].replace("(","").replace(")","").replace("'","")
            run_ids = [e for e in str_list.split(',')]
            print(run_ids)
            return "IN", splits[0], run_ids
        else:
            splits = filter_string.split('=', 1)
            return "=", splits[0], splits[1].replace("'","")
    
    def _search_model_version(self, filter_string):
        op, key, value = self._parse_model_version_string(filter_string=filter_string)
        if op is "=":
            q = Q('match', **{"model_versions."+key: value})
        elif op is "IN":
            q = Q('terms', **{"model_versions."+key:value})
        s = Search(index="mlflow-registered-model").query("nested", path="model_versions", query=q, inner_hits={"size":100})
        response = s.source(['model_versions.name', 'model_versions.version']).execute()
        model_versions = [self._process_inner_hits(hit) for hit in response]
        return [item for sublist in model_versions for item in sublist]

    def search_model_versions(self, filter_string):
        print(filter_string)
        return PagedList(self._search_model_version(filter_string), None)

    def set_model_version_tag(self, name, version, tag):
        last_updated_time = now()
        
        _validate_model_name(name)
        _validate_model_version(version)
        _validate_model_version_tag(tag.key, tag.value)
        
        (elastic_registered_model, elastic_model_version) = self._get_elastic_model_version(name=name,version=version)     
        tags = elastic_model_version.model_version_tags
        tags_dict = {}
        for t in tags or []:
            tags_dict[t.key] = t.value

        # Add/update key in the tags
        tags_dict[tag.key] = tag.value
        model_version_tags = [ElasticModelVersionTag(key=key, value=value) for key, value in tags_dict.items()]
        
        model_versions = []

        for mv in elastic_registered_model.model_versions:
            if mv.version == int(version):
                mv.model_version_tags = model_version_tags
                mv.last_updated_time = last_updated_time
            model_versions.append(mv)
        
        elastic_registered_model.update(refresh=True, model_versions=model_versions, 
                                            last_updated_time=last_updated_time)


    def delete_model_version_tag(self, name, version, key):
        last_updated_time = now()
        
        _validate_model_name(name)
        _validate_model_version(version)
        _validate_tag_name(key)
        
        (elastic_registered_model, elastic_model_version) = self._get_elastic_model_version(name=name,version=version)     
        tags = elastic_model_version.model_version_tags
        tags_dict = {}
        for t in tags or []:
            tags_dict[t.key] = t.value

        # Pop the key if it exists
        tags_dict.pop(key, None)
        model_version_tags = [ElasticModelVersionTag(key=key, value=value) for key, value in tags_dict.items()]
        
        model_versions = []

        for mv in elastic_registered_model.model_versions:
            if mv.version == int(version):
                mv.model_version_tags = model_version_tags
                mv.last_updated_time = last_updated_time
            model_versions.append(mv)
        
        elastic_registered_model.update(refresh=True, model_versions=model_versions, 
                                            last_updated_time=last_updated_time)

