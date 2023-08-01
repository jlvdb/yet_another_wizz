ARG USERNAME=yaw
ARG USER_UID=1000
ARG USER_GID=$USER_UID


# base image: contains all required libraries
FROM python:3.10-slim AS base
# update and install missing libraries
RUN set -eux; \
    apt-get update; \
    apt-get install libgomp1; \
    apt-get autoremove -y; \
    apt-get clean -y; \
    rm -rf /var/lib/apt/lists/*


# dependencies image: virtual environment to (build and) install all dependencies
FROM base AS dependencies
# install tools to build denpendiencies
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        build-essential \
        libffi-dev \
        libbz2-dev
# create the virtual environment
RUN python3 -m venv /venv
ENV PATH=/venv/bin:$PATH
RUN pip install pip-tools
# create a requirements.txt from pyproject.toml
WORKDIR /yaw
COPY pyproject.toml ./
COPY src/yaw/__init__.py src/yaw/
RUN python -m piptools compile \
    -o requirements.txt \
    pyproject.toml
# install the dependencies
RUN pip install -r requirements.txt


# build image: build and install the package
FROM dependencies AS build
# copy required files for build
COPY . /yaw/
# build and install the package
WORKDIR /yaw
RUN pip install .


# release image: pull together all data
FROM base as release
ARG USERNAME
ARG USER_UID
ARG USER_GID
# copy and use the virtual environment
COPY --from=build /venv /venv
ENV PATH=/venv/bin:$PATH
# create a non-root user and add a working directory
RUN set -eux; \
    groupadd --gid ${USER_GID} ${USERNAME}; \
    useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME}
USER ${USER_UID}
WORKDIR /data
