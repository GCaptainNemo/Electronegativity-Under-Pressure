FROM python:3.7

MAINTAINER wanghongyu zy2015213@buaa.edu.com

RUN mkdir -p /workspace

WORKDIR /workspace

COPY . ./
 

RUN pip install -i http://pypi.mirrors.ustc.edu.cn/simple --trusted-host pypi.mirrors.ustc.edu.cn --upgrade pip && \
    pip install -i http://pypi.mirrors.ustc.edu.cn/simple --trusted-host pypi.mirrors.ustc.edu.cn -r /workspace/requirements.txt

RUN cd /workspace && \
    wget https://julialang-s3.julialang.org/bin/linux/x64/1.6/julia-1.6.1-linux-x86_64.tar.gz && \
    tar xvfz julia-1.6.1-linux-x86_64.tar.gz && \
    rm julia-1.6.1-linux-x86_64.tar.gz && \
    cp -r julia-1.6.1 /opt/ && \
    rm -rf /workspace/julia-1.6.1 && \
    ln -s /opt/julia-1.6.1/bin/julia /usr/bin/julia

