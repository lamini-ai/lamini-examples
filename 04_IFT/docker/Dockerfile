FROM python:3.11 as base

ARG PACKAGE_NAME="lamini-ift"

# Install Ubuntu libraries
RUN apt-get -yq update

# Install python packages
WORKDIR /app/${PACKAGE_NAME}
COPY ./requirements.txt /app/${PACKAGE_NAME}/requirements.txt
RUN pip install -r requirements.txt

# Copy all files to the container
COPY ./docker/scripts /app/${PACKAGE_NAME}/scripts
COPY ./generate_data.py /app/${PACKAGE_NAME}/generate_data.py
COPY ./generate_data_pipeline.py /app/${PACKAGE_NAME}/generate_data_pipeline.py
COPY ./train.py /app/${PACKAGE_NAME}/train.py

WORKDIR /app/${PACKAGE_NAME}

RUN chmod a+x /app/${PACKAGE_NAME}/scripts/start.sh

ENV PACKAGE_NAME=$PACKAGE_NAME

ENTRYPOINT ["/app/lamini-ift/scripts/start.sh"]


