"""
RL agent training package.
Run training via: python train_rl_agent.py
"""

from rl_agent.config import (
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
)
from rl_agent.data import load_all_tickers_data
from rl_agent.training import evaluate_on_test_data, train_ppo_agent

__all__ = [
    "EVAL_END_DATE",
    "EVAL_START_DATE",
    "MODEL_SAVE_DIR",
    "TENSORBOARD_LOG_DIR",
    "TICKER_LIST",
    "TOTAL_TIMESTEPS",
    "TRAIN_END_DATE",
    "TRAIN_START_DATE",
    "VAL_END_DATE",
    "VAL_START_DATE",
    "load_all_tickers_data",
    "evaluate_on_test_data",
    "train_ppo_agent",
]
