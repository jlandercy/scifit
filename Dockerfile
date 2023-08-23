FROM python:3.11-bullseye AS builder

LABEL maintainer="jeanlandercy@live.com"

ENV TZ=Etc/UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update \
    && apt-get install -y \
    apt-transport-https \
    build-essential \
    cmake \
    libopenblas-dev \
    nano \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-numpy \
    python3-venv \
    python3-wheel \
    software-properties-common \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Python magic env:
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ENV VIRTUAL_ENV=/opt/venv
# RUN python3 -m venv $VIRTUAL_ENV
# ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --upgrade setuptools virtualenv wheel

COPY ./requirements.txt ./requirements.txt
RUN python3 -m pip install -r ./requirements.txt

FROM builder AS image

RUN apt-get update && \
    apt-get install -y \
    git \
    make \
    packagekit-gtk3-module \
    pandoc \
    gconf-service libasound2 libatk1.0-0 libc6 libcairo2 libcups2 libdbus-1-3 libexpat1 libfontconfig1 \
    libgcc1 libgconf-2-4 libgdk-pixbuf2.0-0 libglib2.0-0 libgtk-3-0 libnspr4 libpango-1.0-0 libpangocairo-1.0-0 \
    libstdc++6 libx11-6 libx11-xcb1 libxcb1 libxcomposite1 libxcursor1 libxdamage1 libxext6 libxfixes3 libxi6 \
    libxrandr2 libxrender1 libxss1 libxtst6 ca-certificates fonts-liberation libnss3 lsb-release \
    xdg-utils wget \
    python3-dbus \
    graphviz graphviz-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

#USER 1000:1000

COPY ./requirements_ci.txt ./requirements_ci.txt
RUN python3 -m pip install -r ./requirements_ci.txt

CMD python3
