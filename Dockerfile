# Base container name
ARG BASE_NAME=python:3.11

FROM $BASE_NAME as base

ARG PACKAGE_NAME="lamini-earnings-sdk"

# Install python packages
WORKDIR /app/${PACKAGE_NAME}
COPY ./requirements.txt /app/${PACKAGE_NAME}/requirements.txt

RUN pip install -r requirements.txt

# Copy all files to the container
COPY scripts /app/${PACKAGE_NAME}/scripts

COPY . /app/${PACKAGE_NAME}
# COPY 02_eval /app/${PACKAGE_NAME}/02_eval
# COPY 03_prompt_tuning /app/${PACKAGE_NAME}/03_prompt_tuning
# COPY 04_rag_tuning /app/${PACKAGE_NAME}/04_rag_tuning
# COPY 05_data_pipeline /app/${PACKAGE_NAME}/05_data_pipeline
# COPY 06_fine_tuning /app/${PACKAGE_NAME}/06_fine_tuning
# COPY utils /app/${PACKAGE_NAME}/utils

WORKDIR /app/${PACKAGE_NAME}

ENV PACKAGE_NAME=$PACKAGE_NAME
ENTRYPOINT ["/bin/bash"]
