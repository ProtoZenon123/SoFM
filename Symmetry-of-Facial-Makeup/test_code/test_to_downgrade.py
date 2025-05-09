from utils import load_config
from evaluate import evaluate
import logging
import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("evaluation.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    # Load configuration using load_config
    config = load_config(config_file_name="exp_config.yaml")

    # Run evaluation
    evaluate(config, logger)