ARG python=python:3.10-slim

# create a container with compilers
FROM ${python} AS base
# upgrade the package index and install security upgrades
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        build-essential \
        libffi-dev \
        libbz2-dev

# create a container to install dependencies into a virtual environment
FROM base AS dependencies
# create the virtual environment
RUN python3 -m venv /venv
ENV PATH=/venv/bin:$PATH
# create a requirements.txt from pyproject.toml with pip-tools
WORKDIR /yaw
RUN pip install pip-tools
COPY pyproject.toml ./
# copy the yaw.__init__ to get the dynamic access to yaw.__version__
COPY src/yaw/__init__.py src/yaw/
RUN python -m piptools compile \
    -o requirements.txt \
    pyproject.toml
# install the dependencies
RUN pip install -r requirements.txt

# create a container to build and install the package
FROM dependencies AS build
WORKDIR /yaw
# copy required files for build
COPY . .
RUN pip install .

# final stage
FROM ${python} as release
# install missing object and clean up
RUN set -eux; \
    apt-get update; \
    apt-get install libgomp1; \
    apt-get autoremove -y; \
    apt-get clean -y; \
    rm -rf /var/lib/apt/lists/*
# copy the virtual environment
COPY --from=build /venv /venv
ENV PATH=/venv/bin:$PATH
# create a non-root user and add a working directory
RUN set -eux; \
    addgroup --system --gid 1001 yaw; \
    adduser --system --no-create-home --uid 1001 --gid 1001 yaw
USER yaw
WORKDIR /data
