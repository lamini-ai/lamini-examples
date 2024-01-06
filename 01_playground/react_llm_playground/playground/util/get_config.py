import config
import os
import yaml

import logging

logger = logging.getLogger(__name__)

global_config = None


def create_config(dictionary_config={}):
    global global_config
    global_config = setup_config(dictionary_config)
    logger.debug("Config: " + str(yaml.dump(global_config.as_dict())))

    setup_logging(global_config)

    return global_config


def get_config():
    global global_config
    assert global_config is not None
    return global_config


def setup_config(dictionary_config):
    config_paths = get_config_paths()
    local_config_path = None
    config_set = None

    # for all llama, llama_edits or any other future named config files.
    for path in config_paths:
        if path.endswith("_local.yaml"):
            local_config_path = path
            continue

        if not config_set:
            config_set = config.config_from_yaml(path, read_from_file=True)
        else:
            config_set.update(config.config_from_yaml(path, read_from_file=True))

    # override with user passed config
    config_set.update(config.config_from_dict(dictionary_config))

    # override with secrets from local machine config
    if local_config_path:
        config_set.update(config.config_from_yaml(path, read_from_file=True))
    return config_set


def get_config_paths():
    paths = []

    config_name = "playground_config"

    config_base = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs")
    logger.debug("Config base: " + str(config_base))

    base_config_path = os.path.join(os.path.dirname(__file__), config_name + "_default.yaml")
    if os.path.exists(base_config_path):
        paths.append(base_config_path)

    edit_config_path = os.path.join(config_base, config_name + "_edits.yaml")
    if os.path.exists(edit_config_path):
        paths.append(edit_config_path)

    local_config_path = os.path.join(config_base, config_name + "_local.yaml")
    if os.path.exists(local_config_path):
        paths.append(local_config_path)

    home = os.path.expanduser("~")
    home_config_path = os.path.join(home, ".playground", config_name + ".yaml")
    if os.path.exists(home_config_path):
        paths.append(home_config_path)

    return paths


def setup_logging(arguments):
    logging_format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"

    if arguments["verbose"]:
        logging.basicConfig(level=logging.DEBUG, format=logging_format)
    elif arguments["verbose_info"]:
        logging.basicConfig(level=logging.INFO, format=logging_format)
    else:
        logging.basicConfig(level=logging.WARNING, format=logging_format)

    root_logger = logging.getLogger()

    if arguments["verbose"]:
        root_logger.setLevel(logging.DEBUG)
    elif arguments["verbose_info"]:
        root_logger.setLevel(logging.INFO)
    else:
        root_logger.setLevel(logging.WARNING)

    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("statsig").setLevel(logging.ERROR)
    logging.getLogger("watchfiles").setLevel(logging.WARNING)

