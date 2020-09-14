#!/bin/bash

docker ps -a
docker images

docker build  --tag elastic_store .
docker run \
    -p 9200:9200 \
    -p 9300:9300 \
    -e "discovery.type=single-node" \
    -e ELASTIC_PASSWORD=password \
    -e xpack.security.enabled=true \
    -v /Users/a.vivien/Workspace/mlflow-elasticsearchstore/tests/snapshot/:/mount/backups/ \
    -d \
    elastic_store