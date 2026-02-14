"""
Train RL Risk Management Agent (RecurrentPPO).
Entry point: run this script to train and evaluate on test data.
All configuration and logic live in the rl_agent package.
"""

import logging
from pathlib import Path

from rl_agent import (
    EVAL_END_DATE,
    EVAL_START_DATE,
    MODEL_SAVE_DIR,
    TENSORBOARD_LOG_DIR,
    TICKER_LIST,
    TOTAL_TIMESTEPS,
    TRAIN_END_DATE,
    TRAIN_START_DATE,
    VAL_END_DATE,
    VAL_START_DATE,
    evaluate_on_test_data,
    load_all_tickers_data,
    train_ppo_agent,
)

Path("logs").mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/train_rl_agent.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def main():
    logger.info("RL Risk Management Agent Training")
    logger.info(
        "Date ranges: train %s–%s | val %s–%s | eval %s–%s",
        TRAIN_START_DATE or "start", TRAIN_END_DATE or "end",
        VAL_START_DATE or "start", VAL_END_DATE or "end",
        EVAL_START_DATE or "start", EVAL_END_DATE or "end",
    )

    train_tickers_data = load_all_tickers_data(
        TICKER_LIST, start_date=TRAIN_START_DATE, end_date=TRAIN_END_DATE
    )
    if len(train_tickers_data) == 0:
        logger.error("No training data loaded. Ensure data/dataset.csv exists.")
        return

    for ticker, data in train_tickers_data.items():
        n_sigs = data["entry_signals"].sum() if data.get("entry_signals") is not None else 0
        logger.info("  %s: %d points, %d signals", ticker, len(data["price"]), n_sigs)

    val_tickers_data = load_all_tickers_data(
        TICKER_LIST, start_date=VAL_START_DATE, end_date=VAL_END_DATE
    )
    if len(val_tickers_data) == 0:
        logger.warning("No validation data; using training data for evaluation.")
        val_tickers_data = train_tickers_data

    test_tickers_data = load_all_tickers_data(
        TICKER_LIST, start_date=EVAL_START_DATE, end_date=EVAL_END_DATE
    )
    if test_tickers_data:
        for ticker, data in test_tickers_data.items():
            n_sigs = data["entry_signals"].sum() if data.get("entry_signals") is not None else 0
            logger.info("  Test %s: %d points, %d signals", ticker, len(data["price"]), n_sigs)

    model = train_ppo_agent(
        train_tickers_data,
        MODEL_SAVE_DIR,
        TENSORBOARD_LOG_DIR,
        total_timesteps=int(TOTAL_TIMESTEPS),
        n_envs=None,
        val_tickers_data=val_tickers_data,
    )

    logger.info("Training pipeline complete. TensorBoard: tensorboard --logdir %s", TENSORBOARD_LOG_DIR)

    if test_tickers_data and model is not None:
        vec_path = MODEL_SAVE_DIR / "vec_normalize.pkl"
        test_results = evaluate_on_test_data(model, test_tickers_data, vec_normalize_path=vec_path)
        if test_results:
            results_file = MODEL_SAVE_DIR / "test_evaluation_results.txt"
            with open(results_file, "w") as f:
                f.write("TEST DATA EVALUATION RESULTS\n")
                f.write("=" * 40 + "\n")
                f.write(f"Test date range: {EVAL_START_DATE} to {EVAL_END_DATE}\n")
                f.write(f"Entry signals evaluated: {test_results['n_episodes']}\n")
                f.write(f"Mean total reward: {test_results['mean_reward']:.2f} +/- {test_results['std_reward']:.2f}\n")
                f.write(f"Mean reward/step: {test_results['mean_reward_per_step']:.4f}\n")
                f.write(f"Win rate: {test_results['win_rate']:.1%}\n")
                f.write(f"Avg holding time: {test_results['avg_holding_time']:.1f} steps\n")
                f.write(f"Total reward: {test_results['total_reward']:.2f}\n")
            logger.info("Test results saved to %s", results_file)
    else:
        logger.warning("Skipping test evaluation (no test data or model).")

    logger.info("Done.")


if __name__ == "__main__":
    main()
