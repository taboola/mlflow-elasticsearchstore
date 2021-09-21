import datetime
from elasticsearch_dsl import (Document, InnerDoc, Nested, Text,
                               Keyword, Double, Integer, Long, Boolean)

from mlflow.entities import (Experiment, RunTag, Metric, Param,
                             RunData, RunInfo, Run, ExperimentTag)

from mlflow.entities.model_registry import (
    RegisteredModel,
    ModelVersion,
    RegisteredModelTag,
    ModelVersionTag,
)

from mlflow.entities.model_registry.model_version_stages import STAGE_NONE, STAGE_DELETED_INTERNAL
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
# TODO look at setting default values 

class ElasticExperimentTag(InnerDoc):
    key = Keyword()
    value = Text()

    def to_mlflow_entity(self) -> ExperimentTag:
        return ExperimentTag(key=self.key,
                             value=self.value)


class ElasticExperiment(Document):
    name = Keyword()
    artifact_location = Text()
    lifecycle_stage = Keyword()
    tags = Nested(ElasticExperimentTag)

    class Index:
        name = 'mlflow-experiments'
        settings = {
            "number_of_shards": 1,
            "number_of_replicas": 1
        }

    def to_mlflow_entity(self) -> Experiment:
        return Experiment(
            experiment_id=str(self.meta.id),
            name=self.name,
            artifact_location=self.artifact_location,
            lifecycle_stage=self.lifecycle_stage,
            tags=[t.to_mlflow_entity() for t in self.tags])


class ElasticMetric(Document):
    key = Keyword()
    value = Double()
    timestamp = Long()
    step = Integer()
    is_nan = Boolean()
    run_id = Keyword()

    class Index:
        name = 'mlflow-metrics'
        settings = {
            "number_of_shards": 2,
            "number_of_replicas": 2
        }

    def to_mlflow_entity(self) -> Metric:
        return Metric(
            key=self.key,
            value=self.value if not self.is_nan else float("nan"),
            timestamp=self.timestamp,
            step=self.step)


class ElasticLatestMetric(InnerDoc):
    key = Keyword()
    value = Double()
    timestamp = Long()
    step = Integer()
    is_nan = Boolean()

    def to_mlflow_entity(self) -> Metric:
        return Metric(
            key=self.key,
            value=self.value if not self.is_nan else float("nan"),
            timestamp=self.timestamp,
            step=self.step)


class ElasticParam(InnerDoc):
    key = Keyword()
    value = Keyword()

    def to_mlflow_entity(self) -> Param:
        return Param(
            key=self.key,
            value=self.value)


class ElasticTag(InnerDoc):
    key = Keyword()
    value = Keyword()

    def to_mlflow_entity(self) -> RunTag:
        return RunTag(
            key=self.key,
            value=self.value)


class ElasticRun(Document):
    run_id = Keyword()
    name = Keyword()
    source_type = Keyword()
    source_name = Keyword()
    experiment_id = Keyword()
    user_id = Keyword()
    status = Keyword()
    start_time = Long()
    end_time = Long()
    source_version = Keyword()
    lifecycle_stage = Keyword()
    artifact_uri = Text()
    latest_metrics = Nested(ElasticLatestMetric)
    params = Nested(ElasticParam)
    tags = Nested(ElasticTag)

    class Index:
        name = 'mlflow-runs'
        settings = {
            "number_of_shards": 2,
            "number_of_replicas": 2
        }

    def to_mlflow_entity(self) -> Run:
        run_info = RunInfo(
            run_uuid=self.meta.id,
            run_id=self.meta.id,
            experiment_id=str(self.experiment_id),
            user_id=self.user_id,
            status=self.status,
            start_time=self.start_time,
            end_time=self.end_time,
            lifecycle_stage=self.lifecycle_stage,
            artifact_uri=self.artifact_uri)

        run_data = RunData(
            metrics=[m.to_mlflow_entity() for m in self.latest_metrics],
            params=[p.to_mlflow_entity() for p in self.params],
            tags=[t.to_mlflow_entity() for t in self.tags])
        return Run(run_info=run_info, run_data=run_data)


class ElasticRegisteredModelTag(InnerDoc):
    key = Keyword()
    value = Text()
    
    def to_mlflow_entity(self):
        return RegisteredModelTag(self.key, self.value)

class ElasticModelVersionTag:
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