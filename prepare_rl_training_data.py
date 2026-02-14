"""
Prepare RL Training Data - Lightweight check before training.
train_rl_agent.py loads data directly from data/dataset.csv with date filters;
no episode extraction is required.
"""

from pathlib import Path
import logging

Path("logs").mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/prepare_rl_training_data.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
DATASET_FILE = DATA_DIR / "dataset.csv"


def main():
    """Verify that dataset exists for train_rl_agent.py (which uses date-based loading)."""
    logger.info("="*60)
    logger.info("RL Training Data Preparation")
    logger.info("="*60)

    if not DATASET_FILE.exists():
        logger.error(f"Dataset not found: {DATASET_FILE}")
        logger.error("Run: python backtest.py --download-only")
        return

    logger.info(f"Dataset found: {DATASET_FILE}")
    logger.info("train_rl_agent.py loads data from dataset.csv with TRAIN_*_DATE / VAL_*_DATE / EVAL_*_DATE.")
    logger.info("No episode extraction needed.")
    logger.info("="*60)
    logger.info("Data preparation complete!")
    logger.info("="*60)


if __name__ == "__main__":
    main()
