"""
Configuration for RL Risk Management Agent training.
Single source of truth for hyperparameters, paths, and date ranges.
"""

import os
from pathlib import Path

# Paths
MODEL_SAVE_DIR = Path("models/rl_agent")
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
TENSORBOARD_LOG_DIR = MODEL_SAVE_DIR / "tensorboard_logs"
TENSORBOARD_LOG_DIR.mkdir(parents=True, exist_ok=True)

# Data
TICKER_LIST = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "DOGEUSDT", "ADAUSDT", "SOLUSDT"]
INITIAL_BALANCE = 1000.0

# Training hyperparameters
TOTAL_TIMESTEPS = 5e6
LEARNING_RATE = 3e-4
LEARNING_RATE_END = 1e-5
USE_LR_SCHEDULE = True
BATCH_SIZE = 1024
N_STEPS = 2048
N_EPOCHS = 4
GAMMA = 0.99
GAE_LAMBDA = 0.95
ENT_COEF = 0.01
VF_COEF = 0.25
MAX_GRAD_NORM = 0.5
CLIP_RANGE = 0.2

# Device
USE_GPU = True

# Parallel training
USE_PARALLEL_ENVS = True
N_ENVS = None
MULTIPROCESSING_START_METHOD = "forkserver" if os.name != "nt" else "spawn"

# Model architecture (RecurrentPPO)
POLICY_LAYERS = [256, 256]
VALUE_LAYERS = [256, 256]
ACTIVATION_FN = "tanh"
LSTM_HIDDEN_SIZE = 256
N_LSTM_LAYERS = 1

# Checkpoints and evaluation
CHECKPOINT_FREQ = 10000
EVAL_FREQ = 1000
N_EVAL_EPISODES = 500

# Early stopping
ENABLE_EARLY_STOPPING = False
EARLY_STOPPING_PATIENCE = 50
EARLY_STOPPING_MIN_DELTA = 0.0
EARLY_STOPPING_MONITOR = "mean_ep_length"
EARLY_STOPPING_MODE = "max"

# Date ranges (YYYY-MM-DD or None)
TRAIN_START_DATE = "2010-01-01"
TRAIN_END_DATE = "2024-01-01"
VAL_START_DATE = "2025-01-01"
VAL_END_DATE = "2026-01-01"
EVAL_START_DATE = "2026-01-01"
EVAL_END_DATE = "2026-02-15"

# Entry/exit signals
USE_TECHNICAL_SIGNALS = True
USE_BACKTEST_SIGNALS = True
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

# Signals cache
SIGNALS_CACHE_DIR = Path("data/signals_cache")
