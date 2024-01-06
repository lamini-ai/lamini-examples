FROM python:3.11 as base

ARG PACKAGE_NAME="chat-playground"

WORKDIR /app/${PACKAGE_NAME}

# Install nodejs
RUN apt-get update && apt-get install -y npm
RUN npm install -g yarn

# install dependencies from package.json
COPY ./package.json /app/${PACKAGE_NAME}/package.json
RUN yarn

# Install python dependencies
COPY ./requirements.txt /app/${PACKAGE_NAME}/requirements.txt
RUN pip install -r requirements.txt

# Install source code
COPY playground /app/${PACKAGE_NAME}/playground
COPY src /app/${PACKAGE_NAME}/src
COPY public /app/${PACKAGE_NAME}/public
COPY scripts /app/${PACKAGE_NAME}/scripts
COPY configs /app/${PACKAGE_NAME}/configs
COPY tsconfig.json /app/${PACKAGE_NAME}/tsconfig.json
COPY postcss.config.js /app/${PACKAGE_NAME}/postcss.config.js
COPY tailwind.config.js /app/${PACKAGE_NAME}/tailwind.config.js

# Setup the entrypoints
RUN chmod a+x /app/${PACKAGE_NAME}/scripts/start.sh
ENV PACKAGE_NAME=$PACKAGE_NAME

ENTRYPOINT /app/${PACKAGE_NAME}/scripts/start.sh

