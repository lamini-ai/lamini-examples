FROM python:3.11 as base

ARG PACKAGE_NAME="lamini-python-client"

# Install Ubuntu libraries
RUN apt-get -yq update

# Install python packages
WORKDIR /app/${PACKAGE_NAME}
COPY ./requirements.txt /app/${PACKAGE_NAME}/requirements.txt
RUN pip install -r requirements.txt

# Copy all files to the container
COPY ./docker/scripts /app/${PACKAGE_NAME}/scripts
COPY ./answer_sample_questions.py /app/${PACKAGE_NAME}/answer_sample_questions.py
COPY ./sample_questions.jsonl /app/${PACKAGE_NAME}/sample_questions.jsonl

WORKDIR /app/${PACKAGE_NAME}

RUN chmod a+x /app/${PACKAGE_NAME}/scripts/start.sh

ENV PACKAGE_NAME=$PACKAGE_NAME

ENTRYPOINT ["/app/lamini-python-client/scripts/start.sh"]

