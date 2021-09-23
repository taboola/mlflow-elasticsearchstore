from elasticsearch_dsl import InnerDoc, Keyword, Text, Long, Document, Nested
from mlflow.entities.model_registry import (
    RegisteredModel,
    ModelVersion,
    RegisteredModelTag,
    ModelVersionTag,
)

from mlflow.entities.model_registry.model_version_stages import STAGE_NONE, STAGE_DELETED_INTERNAL
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
# TODO look at setting default values



class ElasticRegisteredModelTag(InnerDoc):
    key = Keyword()
    value = Text()

    def to_mlflow_entity(self):
        return RegisteredModelTag(self.key, self.value)

class ElasticModelVersionTag(InnerDoc):
    key = Keyword()
    value = Text()

    def to_mlflow_entity(self):
        return ModelVersionTag(self.key, self.value)

class ElasticModelVersion(InnerDoc):
    name = Keyword()
    version = Long()
    creation_time = Long()
    last_updated_time = Long()
    description = Text()
    user_id = Keyword()
    current_stage = Keyword()
    source = Keyword()
    run_id = Keyword()
    run_link = Keyword()
    status = Keyword()
    status_message = Text()
    model_version_tags = Nested(ElasticModelVersionTag)

    def to_mlflow_entity(self):
        return ModelVersion(
            self.name,
            self.version,
            self.creation_time,
            self.last_updated_time,
            self.description,
            self.user_id,
            self.current_stage,
            self.source,
            self.run_id,
            self.status,
            self.status_message,
            [tag.to_mlflow_entity() for tag in self.model_version_tags],
            self.run_link,
        )

class ElasticRegisteredModel(Document):
    name = Keyword()
    creation_time = Long()
    last_updated_time = Long()
    description = Text()
    model_versions = Nested(ElasticModelVersion)
    registered_model_tags = Nested(ElasticRegisteredModelTag)

    class Index:
        name = 'mlflow-registered-model'
        settings = {
            "number_of_shards": 2,
            "number_of_replicas": 2
        }

    def to_mlflow_entity(self):
        latest_versions = {}
        for mv in self.model_versions:
            stage = mv.current_stage
            if stage != STAGE_DELETED_INTERNAL and (
                    stage not in latest_versions or latest_versions[stage].version < mv.version
            ):
                latest_versions[stage] = mv
        return RegisteredModel(
            self.name,
            self.creation_time,
            self.last_updated_time,
            self.description,
            [mvd.to_mlflow_entity() for mvd in latest_versions.values()],
            [tag.to_mlflow_entity() for tag in self.registered_model_tags],
        )