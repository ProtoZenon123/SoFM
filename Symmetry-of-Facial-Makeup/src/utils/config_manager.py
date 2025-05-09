import os
from dotenv import load_dotenv
import yaml

load_dotenv()


class ConfigManager:
    def __init__(
        self, config_dir: str = "../config", config_file_name: str = "exp_config.yaml"
    ):
        self.config_dir = config_dir
        self.config_file_name = config_file_name
        self.config_path = os.path.join(self.config_dir, self.config_file_name)
        self.config = self._load_config()

    def _load_config(self):
        """
        Load the configuration from the config file.
        """
        try:
            with open(self.config_path, "r") as file:
                config = yaml.safe_load(file)
                if not config:
                    raise ValueError("Configuration file is empty.")
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing the configuration file: {e}")

    def get_config(self):
        """
        Get the loaded configuration.
        """
        return self.config


def load_config(config_file_name: str = "exp_config.yaml"):
    """
    Load the configuration from the config file.
    """
    config_dir = os.getenv("CONFIG_DIR")
    config_manager = ConfigManager(
        config_dir=config_dir, config_file_name=config_file_name
    )
    return config_manager.get_config()
