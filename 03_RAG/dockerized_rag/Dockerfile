FROM python:3.11 as base

ARG PACKAGE_NAME="lamini-rag"

# Install Ubuntu libraries
RUN apt-get -yq update

# Install python packages
WORKDIR /app/${PACKAGE_NAME}
COPY ./requirements.txt /app/${PACKAGE_NAME}/requirements.txt
RUN pip install -r requirements.txt

# Copy all files to the container
COPY ./dockerized_rag/scripts /app/${PACKAGE_NAME}/scripts
COPY ./build_index.py /app/${PACKAGE_NAME}/build_index.py
COPY ./lamini_index.py /app/${PACKAGE_NAME}/lamini_index.py

WORKDIR /app/${PACKAGE_NAME}

RUN chmod a+x /app/${PACKAGE_NAME}/scripts/start.sh

ENV PACKAGE_NAME=$PACKAGE_NAME

ENTRYPOINT ["/app/lamini-rag/scripts/start.sh"]


