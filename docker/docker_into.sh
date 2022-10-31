#!/bin/bash
set -e

TOP_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
source $TOP_DIR/docker/config.sh
if [[ $# = 1  ]];then
    CONTAINER_NAME=$1
fi

echo "Into docker container $CONTAINER_NAME"
docker exec  -it $CONTAINER_NAME  /bin/bash
if [ $? -ne 0 ];then
    error "Failed to start docker container"
    exit 1
fi
