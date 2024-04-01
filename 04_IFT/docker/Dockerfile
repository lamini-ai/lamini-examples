FROM python:3.11 as base

ARG PACKAGE_NAME="lamini-ift"

# Install Ubuntu libraries
RUN apt-get -yq update

# Install python packages
WORKDIR /app/${PACKAGE_NAME}
COPY ./04_IFT/requirements.txt /app/${PACKAGE_NAME}/requirements.txt
RUN pip install -r requirements.txt

# Copy all files to the container
COPY ./04_IFT/docker/scripts /app/${PACKAGE_NAME}/scripts
COPY ./04_IFT/generate_data.py /app/${PACKAGE_NAME}/generate_data.py
COPY ./04_IFT/train.py /app/${PACKAGE_NAME}/train.py
COPY ./03_RAG/directory_loader.py /app/${PACKAGE_NAME}/directory_loader.py

WORKDIR /app/${PACKAGE_NAME}

RUN chmod a+x /app/${PACKAGE_NAME}/scripts/start.sh

ENV PACKAGE_NAME=$PACKAGE_NAME

ENTRYPOINT ["/app/lamini-ift/scripts/start.sh"]


