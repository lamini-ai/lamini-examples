FROM python:3.11 as base

ARG PACKAGE_NAME="lamini-classify"

# Install Ubuntu libraries
RUN apt-get -yq update

# Install python packages
WORKDIR /app/${PACKAGE_NAME}
COPY ./requirements.txt /app/${PACKAGE_NAME}/requirements.txt
RUN pip install -r requirements.txt

# Copy all files to the container
COPY ./docker/scripts /app/${PACKAGE_NAME}/scripts
COPY ./classify.py /app/${PACKAGE_NAME}/classify.py
COPY ./calibrate.py /app/${PACKAGE_NAME}/calibrate.py
COPY ./train.py /app/${PACKAGE_NAME}/train.py
COPY ./filter.py /app/${PACKAGE_NAME}/filter.py
COPY ./lamini_classifier.py /app/${PACKAGE_NAME}/lamini_classifier.py

WORKDIR /app/${PACKAGE_NAME}

RUN chmod a+x /app/${PACKAGE_NAME}/scripts/start.sh

ENV PACKAGE_NAME=$PACKAGE_NAME

ENTRYPOINT ["/app/lamini-classify/scripts/start.sh"]

