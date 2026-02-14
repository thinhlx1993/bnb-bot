"""
RL training pipeline: env factory, PPO training, and test evaluation.
"""

import logging
import multiprocessing
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from rl_risk_env import ENV_DEFAULT_CONFIG, RiskManagementEnv
from rl_agent.config import (
    ACTIVATION_FN,
    BATCH_SIZE,
    CHECKPOINT_FREQ,
    CLIP_RANGE,
    ENABLE_EARLY_STOPPING,
    ENT_COEF,
    EVAL_FREQ,
    GAE_LAMBDA,
    GAMMA,
    INITIAL_BALANCE,
    LEARNING_RATE,
    LEARNING_RATE_END,
    LSTM_HIDDEN_SIZE,
    MAX_GRAD_NORM,
    MULTIPROCESSING_START_METHOD,
    N_ENVS,
    N_EPOCHS,
    N_EVAL_EPISODES,
    N_LSTM_LAYERS,
    N_STEPS,
    POLICY_LAYERS,
    TOTAL_TIMESTEPS,
    USE_GPU,
    USE_LR_SCHEDULE,
    USE_PARALLEL_ENVS,
    VALUE_LAYERS,
    EARLY_STOPPING_MIN_DELTA,
    EARLY_STOPPING_MODE,
    EARLY_STOPPING_MONITOR,
    EARLY_STOPPING_PATIENCE,
)
from rl_agent.callbacks import (
    EarlyStoppingCallback,
    SaveVecNormalizeWithBestModel,
    VecNormalizeSyncCallback,
)

logger = logging.getLogger(__name__)


def linear_schedule(initial_lr: float, final_lr: float = 0.0):
    """Linear learning rate schedule: progress_remaining 1.0 -> 0.0."""
    def schedule(progress_remaining: float) -> float:
        return final_lr + progress_remaining * (initial_lr - final_lr)
    return schedule


def _win_rate_from_returns(episode_returns: List[float]) -> Tuple[float, int, int]:
    """Win rate = fraction of episodes with positive return. Returns (win_rate, wins, n_episodes)."""
    if not episode_returns:
        return 0.0, 0, 0
    wins = sum(1 for r in episode_returns if r > 0)
    return wins / len(episode_returns), wins, len(episode_returns)


def _compute_eval_metrics(
    episode_rewards: List[float],
    episode_lengths: List[int],
    episode_returns: Optional[List[float]] = None,
    initial_balance: float = INITIAL_BALANCE,
    failure_count: int = 0,
) -> Dict[str, float]:
    """Compute mean/std reward, win rate, holding time, total balance."""
    if episode_rewards and episode_lengths:
        rewards_per_step = [r / max(l, 1) for r, l in zip(episode_rewards, episode_lengths)]
        mean_reward = float(np.mean(rewards_per_step))
        std_reward = float(np.std(rewards_per_step))
        mean_total_reward = float(np.mean(episode_rewards))
        std_total_reward = float(np.std(episode_rewards))
        win_rate = _win_rate_from_returns(episode_returns or [])[0]
        total_balance = initial_balance + sum(episode_returns or []) * initial_balance
        avg_holding_time = float(np.mean(episode_lengths))
        std_holding_time = float(np.std(episode_lengths))
        min_holding_time = int(np.min(episode_lengths))
        max_holding_time = int(np.max(episode_lengths))
        mean_ep_length = float(np.mean(episode_lengths))
    else:
        mean_reward = std_reward = mean_total_reward = std_total_reward = 0.0
        win_rate = avg_holding_time = std_holding_time = mean_ep_length = 0.0
        min_holding_time = max_holding_time = 0
        total_balance = initial_balance
    return {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "mean_total_reward": mean_total_reward,
        "std_total_reward": std_total_reward,
        "win_rate": win_rate,
        "mean_ep_length": mean_ep_length,
        "avg_holding_time": avg_holding_time,
        "std_holding_time": std_holding_time,
        "min_holding_time": min_holding_time,
        "max_holding_time": max_holding_time,
        "total_balance": total_balance,
        "failure_count": failure_count,
    }


def create_env_factory(all_tickers_data: Dict, seed: int = 42):
    """Create environment factory that uses all tickers' data; on reset, selects ticker and entry."""
    np.random.seed(seed)

    def _make_env(rank: int = 0):
        env = RiskManagementEnv(
            all_tickers_data=all_tickers_data,
            initial_balance=INITIAL_BALANCE,
            **ENV_DEFAULT_CONFIG,
        )
        env = Monitor(env, filename=None, allow_early_resets=True)
        env.reset(seed=seed + rank)
        return env

    return _make_env


def train_ppo_agent(
    all_tickers_data: Dict,
    model_save_dir: Path,
    tensorboard_log_dir: Path,
    total_timesteps: Optional[int] = None,
    n_envs: Optional[int] = None,
    val_tickers_data: Optional[Dict] = None,
):
    """Train RecurrentPPO on all tickers' data with parallel envs and VecNormalize."""
    if total_timesteps is None:
        total_timesteps = int(TOTAL_TIMESTEPS)
    if val_tickers_data is None:
        val_tickers_data = all_tickers_data

    if n_envs is None:
        n_envs = N_ENVS if N_ENVS is not None else max(1, (os.cpu_count() or 4) - 1)
        if N_ENVS is None:
            logger.info("Auto-detected CPU count: using %d parallel environments", n_envs)

    logger.info("Training PPO Agent (RecurrentPPO)")
    try:
        multiprocessing.set_start_method(MULTIPROCESSING_START_METHOD, force=True)
    except RuntimeError:
        pass

    vec_env_cls = SubprocVecEnv if USE_PARALLEL_ENVS else DummyVecEnv
    env_type = "parallel (SubprocVecEnv)" if USE_PARALLEL_ENVS else "sequential (DummyVecEnv)"
    logger.info("Using %s with %d environments", env_type, n_envs)

    train_env = make_vec_env(
        create_env_factory(all_tickers_data, seed=42),
        n_envs=n_envs,
        vec_env_cls=vec_env_cls,
    )
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        training=True,
    )

    eval_env = make_vec_env(
        create_env_factory(val_tickers_data, seed=123),
        n_envs=1,
        vec_env_cls=vec_env_cls,
    )
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        training=False,
    )

    import torch
    import torch.nn as nn
    device = "cuda" if (USE_GPU and torch.cuda.is_available()) else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    activation_map = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}
    activation_fn = activation_map.get(ACTIVATION_FN.lower(), nn.Tanh)
    policy_kwargs = dict(
        net_arch={"pi": POLICY_LAYERS, "vf": VALUE_LAYERS},
        activation_fn=activation_fn,
        lstm_hidden_size=LSTM_HIDDEN_SIZE,
        n_lstm_layers=N_LSTM_LAYERS,
        enable_critic_lstm=True,
        shared_lstm=False,
    )
    lr = linear_schedule(LEARNING_RATE, LEARNING_RATE_END) if USE_LR_SCHEDULE else LEARNING_RATE

    model = RecurrentPPO(
        "MlpLstmPolicy",
        train_env,
        learning_rate=lr,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        ent_coef=ENT_COEF,
        vf_coef=VF_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        clip_range=CLIP_RANGE,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(tensorboard_log_dir),
        verbose=0,
        device=device,
    )

    callbacks = [
        CheckpointCallback(
            save_freq=CHECKPOINT_FREQ,
            save_path=str(model_save_dir / "checkpoints"),
            name_prefix="ppo_risk_agent",
            save_replay_buffer=True,
            save_vecnormalize=True,
        ),
        VecNormalizeSyncCallback(train_env, eval_env, verbose=1),
        EvalCallback(
            eval_env,
            best_model_save_path=str(model_save_dir / "best_model"),
            log_path=str(model_save_dir / "eval_logs"),
            eval_freq=EVAL_FREQ,
            deterministic=True,
            render=False,
            n_eval_episodes=N_EVAL_EPISODES,
            verbose=1,
        ),
    ]
    eval_callback = callbacks[-1]
    callbacks.append(
        SaveVecNormalizeWithBestModel(
            train_env, eval_callback,
            best_model_save_path=str(model_save_dir / "best_model"),
            verbose=1,
        )
    )
    if ENABLE_EARLY_STOPPING:
        callbacks.append(
            EarlyStoppingCallback(
                eval_callback=eval_callback,
                patience=EARLY_STOPPING_PATIENCE,
                min_delta=EARLY_STOPPING_MIN_DELTA,
                monitor=EARLY_STOPPING_MONITOR,
                mode=EARLY_STOPPING_MODE,
                verbose=1,
            )
        )
    callback_list = CallbackList(callbacks)

    logger.info(
        "Starting training for %s timesteps; checkpoints every %s; eval every %s",
        total_timesteps, CHECKPOINT_FREQ, EVAL_FREQ,
    )
    try:
        model.learn(total_timesteps=total_timesteps, callback=callback_list, progress_bar=True)
        model.save(str(model_save_dir / "ppo_risk_agent_final.zip"))
        train_env.save(str(model_save_dir / "vec_normalize.pkl"))
        logger.info("Training complete. Final model and VecNormalize saved.")
    except KeyboardInterrupt:
        model.save(str(model_save_dir / "ppo_risk_agent_interrupted.zip"))
        train_env.save(str(model_save_dir / "vec_normalize.pkl"))
        logger.info("Training interrupted; model and VecNormalize saved.")

    train_env.close()
    eval_env.close()
    return model


def _evaluate_with_episode_returns(model, vec_env, n_eval_episodes: int, deterministic: bool = True):
    """Run evaluation and return (episode_rewards, episode_lengths, episode_returns). Supports LSTM."""
    episode_rewards = []
    episode_lengths = []
    episode_returns = []
    episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
    reset_result = vec_env.reset()
    obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    step_count = np.zeros(vec_env.num_envs, dtype=int)
    total_rewards = np.zeros(vec_env.num_envs)
    lstm_states = None
    n_completed = 0

    while n_completed < n_eval_episodes:
        actions, lstm_states = model.predict(
            obs, state=lstm_states, episode_start=episode_starts, deterministic=deterministic
        )
        obs, rewards, dones, infos = vec_env.step(actions)
        total_rewards += rewards
        step_count += 1
        episode_starts = dones
        for i in range(vec_env.num_envs):
            if dones[i]:
                episode_rewards.append(float(total_rewards[i]))
                episode_lengths.append(int(step_count[i]))
                episode_returns.append(float(infos[i].get("episode_return", 0.0)))
                total_rewards[i] = 0.0
                step_count[i] = 0
                n_completed += 1
    return episode_rewards, episode_lengths, episode_returns


def evaluate_on_test_data(
    model,
    test_tickers_data: Dict,
    vec_normalize_path: Optional[Path] = None,
) -> Dict:
    """Evaluate the trained model on all entry signals in test data."""
    logger.info("Evaluating on test data")
    if len(test_tickers_data) == 0:
        return {}
    total_signals = sum(
        (data["entry_signals"].sum() if data.get("entry_signals") is not None else 0)
        for data in test_tickers_data.values()
    )
    if total_signals == 0:
        logger.warning("No entry signals in test data")
        return {}
    n_episodes = total_signals

    test_env = make_vec_env(
        create_env_factory(test_tickers_data, seed=456),
        n_envs=min(n_episodes, 32),
        vec_env_cls=DummyVecEnv,
    )
    test_env = VecNormalize(test_env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)
    if vec_normalize_path is not None and vec_normalize_path.exists():
        test_env = VecNormalize.load(str(vec_normalize_path), test_env)

    episode_rewards, episode_lengths, episode_returns = _evaluate_with_episode_returns(
        model, test_env, n_eval_episodes=n_episodes, deterministic=True
    )
    test_env.close()

    rewards_per_step = [r / max(l, 1) for r, l in zip(episode_rewards, episode_lengths)]
    win_rate, wins, n_wr = _win_rate_from_returns(episode_returns)
    return {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_reward_per_step": float(np.mean(rewards_per_step)),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "win_rate": win_rate,
        "total_reward": sum(episode_rewards),
        "n_episodes": len(episode_rewards),
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "avg_holding_time": float(np.mean(episode_lengths)),
        "std_holding_time": float(np.std(episode_lengths)),
        "min_holding_time": int(np.min(episode_lengths)),
        "max_holding_time": int(np.max(episode_lengths)),
        "episode_returns": episode_returns,
    }
