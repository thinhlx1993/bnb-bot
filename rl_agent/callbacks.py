"""
Training callbacks: early stopping and VecNormalize sync/save.
"""

import logging
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from rl_agent.config import (
    EARLY_STOPPING_MIN_DELTA,
    EARLY_STOPPING_MODE,
    EARLY_STOPPING_MONITOR,
    EARLY_STOPPING_PATIENCE,
)

logger = logging.getLogger(__name__)


class EarlyStoppingCallback(BaseCallback):
    """
    Stops training if the monitored metric does not improve for a given number of evaluations.
    """

    def __init__(
        self,
        eval_callback,
        patience: int = EARLY_STOPPING_PATIENCE,
        min_delta: float = EARLY_STOPPING_MIN_DELTA,
        monitor: str = EARLY_STOPPING_MONITOR,
        mode: str = EARLY_STOPPING_MODE,
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)
        self.eval_callback = eval_callback
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.best_value = float("-inf") if mode == "max" else float("inf")
        self.wait = 0
        self.stopped_epoch = 0
        self.last_mean_reward = None

        logger.info(
            "Early stopping: monitor=%s, mode=%s, patience=%s, min_delta=%s",
            monitor, mode, patience, min_delta,
        )

    def _on_step(self) -> bool:
        if not hasattr(self.eval_callback, "last_mean_reward"):
            return True
        current_mean_reward = self.eval_callback.last_mean_reward
        if self.last_mean_reward is not None and current_mean_reward == self.last_mean_reward:
            return True
        self.last_mean_reward = current_mean_reward

        current_value = self._get_monitor_value()
        if current_value is None:
            return True
        try:
            current_value = float(current_value)
            if not np.isfinite(current_value):
                return True
        except (TypeError, ValueError):
            return True

        check_mode = "min" if self.monitor == "loss" else self.mode
        if check_mode == "max":
            improved = current_value >= self.best_value + self.min_delta
        else:
            improved = current_value <= self.best_value - self.min_delta

        if improved:
            self.best_value = current_value
            self.wait = 0
            if self.verbose >= 1:
                logger.info(
                    "Early stopping: %s improved to %.4f (best: %.4f)",
                    self.monitor, current_value, self.best_value,
                )
        else:
            self.wait += 1
            if self.verbose >= 1:
                logger.info(
                    "Early stopping: %s no improvement. Wait: %d/%d (current: %.4f, best: %.4f)",
                    self.monitor, self.wait, self.patience, current_value, self.best_value,
                )

        if self.wait >= self.patience:
            self.stopped_epoch = self.num_timesteps
            logger.info(
                "Early stopping triggered at %s timesteps. Best %s: %.4f",
                self.num_timesteps, self.monitor, self.best_value,
            )
            return False
        return True

    def _get_monitor_value(self):
        if self.monitor == "mean_reward":
            return getattr(self.eval_callback, "last_mean_reward", None)
        if self.monitor == "mean_ep_length":
            return getattr(self.eval_callback, "last_mean_ep_length", None)
        if self.monitor == "mean_total_reward":
            return getattr(self.eval_callback, "last_mean_total_reward", None)
        if self.monitor == "win_rate":
            return getattr(self.eval_callback, "last_win_rate", None)
        if self.monitor == "loss":
            r = getattr(self.eval_callback, "last_mean_reward", None)
            return -float(r) if r is not None else None
        return getattr(self.eval_callback, "last_mean_reward", None)


class VecNormalizeSyncCallback(BaseCallback):
    """Syncs VecNormalize statistics from training env to eval env before evaluation."""

    def __init__(self, train_env, eval_env, verbose=0):
        super().__init__(verbose=verbose)
        self.train_env = train_env
        self.eval_env = eval_env

    def _on_step(self) -> bool:
        if hasattr(self.train_env, "obs_rms") and hasattr(self.eval_env, "obs_rms"):
            self.eval_env.obs_rms.mean = self.train_env.obs_rms.mean.copy()
            self.eval_env.obs_rms.var = self.train_env.obs_rms.var.copy()
            self.eval_env.obs_rms.count = self.train_env.obs_rms.count
        return True


class SaveVecNormalizeWithBestModel(BaseCallback):
    """Saves VecNormalize statistics when EvalCallback saves a new best model."""

    def __init__(self, train_env, eval_callback, best_model_save_path, verbose=0):
        super().__init__(verbose=verbose)
        self.train_env = train_env
        self.eval_callback = eval_callback
        self.best_model_save_path = Path(best_model_save_path)
        self.last_best_mean_reward = float("-inf")

    def _on_step(self) -> bool:
        if not hasattr(self.eval_callback, "best_mean_reward"):
            return True
        current_best = self.eval_callback.best_mean_reward
        if current_best > self.last_best_mean_reward:
            self.last_best_mean_reward = current_best
            path = self.best_model_save_path / "vec_normalize.pkl"
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                self.train_env.save(str(path))
                if self.verbose >= 1:
                    logger.info(
                        "VecNormalize saved with best model (mean_reward: %.2f) -> %s",
                        current_best, path,
                    )
            except Exception as e:
                logger.warning("Could not save VecNormalize with best model: %s", e)
        return True
