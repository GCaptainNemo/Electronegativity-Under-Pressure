#!/usr/bin/env bash

REPO_URL=docker.io
DOCKER_IMAGE_NAME=gcaptainnemo/python-install:update
DOCKER_RUNNING_DOCKER_ENV=$REPO_URL/$DOCKER_IMAGE_NAME
CONTAINER_NAME="PYTHON_DAN"
CODE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
