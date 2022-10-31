#!/bin/bash

function install_dependencies() {
    # install dependencies of docker-ce
    sudo apt-get update -y
    sudo apt-get remove -y docker \
               docker-engine \
               docker.io
    sudo apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    software-properties-common

    # add apt source of docker-ce
    curl -fsSL https://mirrors.ustc.edu.cn/docker-ce/linux/ubuntu/gpg | sudo apt-key add -
}

function install_nvidia_docker_runtime() {
    sudo docker volume ls -q -f driver=nvidia-docker | xargs -r -I{} -n1 docker ps -q -a -f volume={} | xargs -r docker rm -f
    sudo apt-get purge -y nvidia-docker

    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
    sudo apt-key add -
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
        sudo tee /etc/apt/sources.list.d/nvidia-docker.list
    sudo apt-get update

    # Install nvidia-docker2 and reload the Docker daemon configuration
    sudo apt-get install -y nvidia-docker2
    sudo pkill -SIGHUP dockerd
}

function install_docker_x86() {
    sudo add-apt-repository \
        "deb [arch=amd64] https://mirrors.ustc.edu.cn/docker-ce/linux/ubuntu \
        $(lsb_release -cs) \
        stable"
    # install docker-ce
    sudo apt-get update -y && sudo apt-get install -y docker-ce
    # install_nvidia_docker_runtime
}

function install_docker_arm() {
    sudo add-apt-repository \
        "deb [arch=armhf] https://mirrors.ustc.edu.cn/docker-ce/linux/ubuntu \
        $(lsb_release -cs) \
        stable"
    # install docker-ce
    sudo apt-get update -y && sudo apt-get install -y docker-ce
}

function docker_config() {
    # setting for docker-ce
    sudo systemctl enable docker
    sudo systemctl start docker
    sudo groupadd docker
    sudo usermod -aG docker $USER

    # setting docker config
    if [ "$MACHINE_ARCH" == 'x86_64' ]; then
      sudo bash -c 'cat <<EOF > /etc/docker/daemon.json
{
    "runtimes": {
        "nvidia": {
            "path": "/usr/bin/nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "registry-mirrors": [
        "https://registry.docker-cn.com"
    ],
}
EOF'
    elif [ "$MACHINE_ARCH" == 'aarch64' ]; then
      sudo bash -c 'cat <<EOF > /etc/docker/daemon.json
{
    "runtimes": {
        "nvidia": {
            "path": "/usr/bin/nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "registry-mirrors": [
        "https://registry.docker-cn.com"
    ],
}
EOF'
    else
      echo "Unknown machine architecture $MACHINE_ARCH"
      exit 1
    fi

    sudo service docker restart
}


# the machine type, currently support x86_64, aarch64
MACHINE_ARCH=$(uname -m)

install_dependencies

if [ "$MACHINE_ARCH" == 'x86_64' ]; then
  install_docker_x86
elif [ "$MACHINE_ARCH" == 'aarch64' ]; then
  install_docker_arm
else
  echo "Unknown machine architecture $MACHINE_ARCH"
  exit 1
fi

install_nvidia_docker_runtime
docker_config
