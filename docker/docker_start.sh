#!/bin/bash

TOP_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
source $TOP_DIR/docker/config.sh

if [[ $# = 1  ]];then
    CONTAINER_NAME=$1
fi
echo "in docker CODE_DIR: ${CODE_DIR}"



function start_docker() {
    # echo "Start login hub.docker.com"
    # docker login
    echo "Start pulling docker image $DOCKER_RUNNING_DOCKER_ENV ..."
    docker pull $DOCKER_RUNNING_DOCKER_ENV

    filter_name="name=^$CONTAINER_NAME\$"
    if [ -n "$(docker container ls -aq -f=${filter_name})" ]; then
        docker stop $CONTAINER_NAME >/dev/null 2>&1
        docker rm -f $CONTAINER_NAME >/dev/null 2>&1
    fi
    echo "Start to run docker container ${CONTAINER_NAME} ..."
    docker run -it \
               -d \
               -p 8888:8888\
               --name ${CONTAINER_NAME} \
               -v $CODE_DIR:/workspace/ \
               $DOCKER_RUNNING_DOCKER_ENV \
               /bin/bash
    echo "finished to run docker container..."
    docker ps |grep $CONTAINER_NAME
    if [ $? -ne 0 ]; then
        echo "Failed to start docker container \"${CONTAINER_NAME}\" based on image: $DOCKER_RUNNING_DOCKER_ENV"
        exit 100
    fi
}

function main() {
    start_docker
}

main
