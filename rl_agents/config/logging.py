import logging.config
import os
from pathlib import Path
import yaml


def setup_logger(experiment_name: str, config_file_name="logging_config.yaml") -> None:
    config_file_path = Path(__file__).parent / config_file_name
    with open(config_file_path) as f:
        config = yaml.safe_load(f)

    log_file = os.path.join("logs", experiment_name, "app.log")

    config["handlers"]["file"]["filename"] = str(log_file)
    logging.config.dictConfig(config)
