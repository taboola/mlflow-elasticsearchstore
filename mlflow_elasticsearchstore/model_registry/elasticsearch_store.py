import time
import logging

from elasticsearch_dsl import Search, connections, Q
from elasticsearch.exceptions import NotFoundError

from six.moves import urllib
from typing import List, Tuple, Any, Dict

from mlflow.entities.model_registry.model_version_stages import (
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
from mlflow_elasticsearchstore.models import (ElasticRegisteredModel, ElasticRegisteredModelTag,
                                              ElasticModelVersion, ElasticModelVersionTag)
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
        connections.create_connection(hosts=[urllib.parse.urlparse(store_uri).netloc])
        ElasticRegisteredModel.init()
        super(ElasticsearchStore, self).__init__()

    def _hit_to_mlflow_registered_model(self, hit) -> RegisteredModel:
        # todo parse other types
        return RegisteredModel(name=hit.name,
                    description=hit.description)

    def _get_registered_model(self, name) -> ElasticRegisteredModel:
        try:
            registered_model = ElasticRegisteredModel.get(name=name)
        except NotFoundError:
            raise MlflowException(
                "No Experiment with name={} exists".format(name), RESOURCE_DOES_NOT_EXIST
            )
        return registered_model
    
    def _registered_model_exists(self, name) -> bool:
        try:
            ElasticRegisteredModel.get(name=name)
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
                    "Registered Model (name={}) already exists. " "Error: {}".format(name),
                    RESOURCE_ALREADY_EXISTS,
            )

        tags_dict = {}
        for tag in tags:
            tags_dict[tag.key] = tag.value

        registered_model_tags = [ElasticRegisteredModelTag(key=key, value=value) for key, value in tags_dict.items()]
        registered_model = ElasticRegisteredModel(
            name=name,
            creation_time=creation_time,
            last_updated_time=creation_time,
            description=description,
            registered_model_tags=registered_model_tags
        )
        registered_model.save()
        return registered_model.to_mlflow_entity()
    
    def update_registered_model(self, name, description) -> RegisteredModel:
        updated_time = now()
        registered_model = self._get_registered_model(name)
        registered_model.update(refresh=True, description=description, last_updated_time=updated_time)
        return registered_model.to_mlflow_entity()
    
    def rename_registered_model(self, name, new_name) -> RegisteredModel:
        updated_time = now()
        registered_model = self._get_registered_model(name)
        registered_model.update(refresh=True, name=new_name, last_updated_time=updated_time)
        return registered_model.to_mlflow_entity()

    def delete_registered_model(self, name) -> None:
        registered_model = self._get_registered_model(name)
        registered_model.delete()

    def list_registered_models(self, max_results, page_token) -> List[RegisteredModel]:
        return self.search_registered_models(max_results=max_results, page_token=page_token)

    def search_registered_models(self, filter_string=None, max_results=None, order_by=None, page_token=None):
        pass

    def get_registered_model(self, name) -> RegisteredModel:
        return self._get_registered_model(name).to_mlflow_entity()

    def get_latest_versions(self, name, stages=None) -> List[ModelVersion]:
        registered_model = self.get_registered_model(name)
        latest_versions = registered_model.to_mlflow_entity().latest_versions
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
        
        registered_model = self.get_registered_model(name)

        tags = registered_model.to_mlflow_entity().tags
        tags_dict = {}
        for t in tags:
            tags_dict[t.key] = t.value

        # Add/update key in the tags
        tags_dict[tag.key] = tag.value
        registered_model_tags = [ElasticRegisteredModelTag(key=key, value=value) for key, value in tags_dict.items()]
        registered_model.update(refresh=True, registered_model_tags=registered_model_tags)

    def delete_registered_model_tag(self, name, key):
        _validate_model_name(name)
        registered_model = self.get_registered_model(name)
        tags = registered_model.to_mlflow_entity().tags
        tags_dict = {}
        for tag in tags:
            tags_dict[tag.key] = tag.value

        # Delete tag if it exists
        tags.pop(key, None)
        
        registered_model_tags = [ElasticRegisteredModelTag(key=key, value=value) for key, value in tags_dict.items()]
        registered_model.update(refresh=True, registered_model_tags=registered_model_tags)

    def create_model_version(self, name, source, run_id=None, tags=None, run_link=None, description=None):
        pass

    def update_model_version(self, name, version, description):
        pass

    def transition_model_version_stage(self, name, version, stage, archive_existing_versions):
        pass

    def delete_model_version(self, name, version):
        pass

    def get_model_version(self, name, version):
        pass

    def get_model_version_download_uri(self, name, version):
        pass

    def search_model_versions(self, filter_string):
        pass

    def set_model_version_tag(self, name, version, tag):
        pass

    def delete_model_version_tag(self, name, version, key):
        pass