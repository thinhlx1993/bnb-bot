"""
Train RL Risk Management Agent using Stable-Baselines3 PPO
"""

import os
import sys
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# RL imports
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList
)
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

# Custom environment
from rl_risk_env import RiskManagementEnv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('train_rl_agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
TRAINING_DATA_DIR = Path("rl_training_episodes")
MODEL_SAVE_DIR = Path("models/rl_agent")
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
TENSORBOARD_LOG_DIR = MODEL_SAVE_DIR / "tensorboard_logs"
TENSORBOARD_LOG_DIR.mkdir(parents=True, exist_ok=True)

# Training hyperparameters
TOTAL_TIMESTEPS = 500000  # Total training steps
LEARNING_RATE = 3e-4
BATCH_SIZE = 64
N_STEPS = 2048  # Steps per update
N_EPOCHS = 10  # Optimization epochs per update
GAMMA = 0.99  # Discount factor
GAE_LAMBDA = 0.95  # GAE lambda
ENT_COEF = 0.01  # Entropy coefficient (exploration)
VF_COEF = 0.5  # Value function coefficient
MAX_GRAD_NORM = 0.5  # Maximum gradient norm for clipping
CHECKPOINT_FREQ = 10000  # Save checkpoint every N steps
EVAL_FREQ = 5000  # Evaluate every N steps
EVAL_EPISODES = 10  # Number of episodes for evaluation

# Training/Validation split
TRAIN_SPLIT = 0.8  # 80% training, 20% validation


def load_episodes(data_dir: Path) -> List[Dict]:
    """
    Load training episodes from disk.
    
    Args:
        data_dir: Directory containing episodes.pkl
    
    Returns:
        List of episode dictionaries
    """
    logger.info(f"Loading episodes from {data_dir}...")
    
    episodes_file = data_dir / "episodes.pkl"
    if not episodes_file.exists():
        logger.error(f"Episodes file not found: {episodes_file}")
        logger.error("Please run prepare_rl_training_data.py first!")
        return []
    
    with open(episodes_file, 'rb') as f:
        episodes = pickle.load(f)
    
    logger.info(f"Loaded {len(episodes)} episodes")
    return episodes


def split_episodes(episodes: List[Dict], train_split: float = 0.8) -> Tuple[List[Dict], List[Dict]]:
    """
    Split episodes into training and validation sets.
    
    Args:
        episodes: List of all episodes
        train_split: Proportion for training (rest for validation)
    
    Returns:
        train_episodes, val_episodes
    """
    # Simple random split (could also do time-based split)
    np.random.seed(42)
    indices = np.random.permutation(len(episodes))
    split_idx = int(len(episodes) * train_split)
    
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_episodes = [episodes[i] for i in train_indices]
    val_episodes = [episodes[i] for i in val_indices]
    
    logger.info(f"Split episodes: {len(train_episodes)} training, {len(val_episodes)} validation")
    
    return train_episodes, val_episodes


def make_env(episode: Dict, rank: int = 0, seed: int = 0) -> RiskManagementEnv:
    """
    Create a single environment instance.
    
    Args:
        episode: Episode dictionary with price_data, balance_data, etc.
        rank: Environment rank (for vectorization)
        seed: Random seed
    
    Returns:
        Environment instance
    """
    def _init():
        env = RiskManagementEnv(
            price_data=episode['price_series'],
            balance_data=episode['balance_series'],
            entry_price=episode['entry_price'],
            entry_idx=episode['entry_idx'] - episode['entry_idx'],  # Relative to episode start
            exit_idx=episode['exit_idx'] - episode['entry_idx'],
            initial_balance=episode['initial_balance'],
            history_length=30,
            max_steps=min(episode['episode_length'] + 100, 1000),
            fee_rate=0.001
        )
        # Wrap with Monitor for statistics
        env = Monitor(env, filename=None, allow_early_resets=True)
        env.reset(seed=seed + rank)
        return env
    return _init


def create_env_factory(train_episodes: List[Dict], seed: int = 42) -> callable:
    """
    Create environment factory that samples random episodes.
    
    Args:
        train_episodes: List of training episodes
        seed: Random seed
    
    Returns:
        Function that returns an environment
    """
    np.random.seed(seed)
    episode_counter = [0]  # Use list to allow modification in closure
    
    def _make_env(rank: int = 0):
        # Cycle through episodes with some randomness
        if np.random.random() < 0.5:
            # Random sampling
            episode = train_episodes[np.random.randint(len(train_episodes))]
        else:
            # Sequential cycling
            episode = train_episodes[episode_counter[0] % len(train_episodes)]
            episode_counter[0] += 1
        return make_env(episode, rank=rank, seed=seed + rank)()
    
    return _make_env


def create_eval_env_factory(val_episodes: List[Dict], seed: int = 42) -> callable:
    """
    Create evaluation environment factory.
    
    Args:
        val_episodes: List of validation episodes
        seed: Random seed
    
    Returns:
        Function that returns an environment
    """
    np.random.seed(seed)
    val_episode_idx = 0
    
    def _make_env(rank: int = 0):
        nonlocal val_episode_idx
        # Cycle through validation episodes
        episode = val_episodes[val_episode_idx % len(val_episodes)]
        val_episode_idx += 1
        return make_env(episode, rank=rank, seed=seed)()
    
    return _make_env


def train_ppo_agent(
    train_episodes: List[Dict],
    val_episodes: List[Dict],
    model_save_dir: Path,
    tensorboard_log_dir: Path,
    total_timesteps: int = TOTAL_TIMESTEPS,
    n_envs: int = 4
):
    """
    Train PPO agent on episodes.
    
    Args:
        train_episodes: Training episodes
        val_episodes: Validation episodes
        model_save_dir: Directory to save models
        tensorboard_log_dir: Directory for TensorBoard logs
        total_timesteps: Total training steps
        n_envs: Number of parallel environments
    """
    logger.info("="*60)
    logger.info("Training PPO Agent")
    logger.info("="*60)
    
    # Create vectorized environments
    logger.info(f"Creating {n_envs} parallel environments...")
    train_env_factory = create_env_factory(train_episodes, seed=42)
    train_env = make_vec_env(
        train_env_factory,
        n_envs=n_envs,
        vec_env_cls=DummyVecEnv  # Use DummyVecEnv for simplicity
    )
    
    # Create evaluation environment
    eval_env_factory = create_eval_env_factory(val_episodes, seed=123)
    eval_env = make_vec_env(
        eval_env_factory,
        n_envs=1,
        vec_env_cls=DummyVecEnv
    )
    
    # Create PPO model
    logger.info("Creating PPO model...")
    policy_kwargs = dict(
        net_arch=[dict(pi=[256, 256], vf=[256, 256])]  # Two hidden layers, 256 units each
    )
    
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        ent_coef=ENT_COEF,
        vf_coef=VF_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(tensorboard_log_dir),
        verbose=1,
        device='cpu'  # Use 'cuda' if GPU available
    )
    
    logger.info(f"Model architecture:")
    logger.info(f"  Policy network: {policy_kwargs['net_arch']}")
    logger.info(f"  Learning rate: {LEARNING_RATE}")
    logger.info(f"  Batch size: {BATCH_SIZE}")
    logger.info(f"  Steps per update: {N_STEPS}")
    logger.info(f"  Epochs per update: {N_EPOCHS}")
    
    # Create callbacks
    callbacks = []
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=str(model_save_dir / "checkpoints"),
        name_prefix="ppo_risk_agent",
        save_replay_buffer=True,
        save_vecnormalize=True
    )
    callbacks.append(checkpoint_callback)
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_save_dir),
        log_path=str(model_save_dir / "eval_logs"),
        eval_freq=EVAL_FREQ,
        n_eval_episodes=EVAL_EPISODES,
        deterministic=True,
        render=False
    )
    callbacks.append(eval_callback)
    
    callback_list = CallbackList(callbacks)
    
    # Train model
    logger.info(f"\nStarting training for {total_timesteps} timesteps...")
    logger.info(f"Checkpoints will be saved every {CHECKPOINT_FREQ} steps")
    logger.info(f"Evaluation every {EVAL_FREQ} steps")
    logger.info(f"TensorBoard logs: {tensorboard_log_dir}")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback_list,
            progress_bar=True
        )
        
        # Save final model
        final_model_path = model_save_dir / "ppo_risk_agent_final.zip"
        model.save(str(final_model_path))
        logger.info(f"\nFinal model saved: {final_model_path}")
        
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        final_model_path = model_save_dir / "ppo_risk_agent_interrupted.zip"
        model.save(str(final_model_path))
        logger.info(f"Model saved: {final_model_path}")
    
    # Cleanup
    train_env.close()
    eval_env.close()
    
    logger.info("="*60)
    logger.info("Training complete!")
    logger.info(f"Model saved to: {model_save_dir}")
    logger.info(f"TensorBoard logs: {tensorboard_log_dir}")
    logger.info("="*60)
    
    return model


def main():
    """Main execution function."""
    logger.info("="*60)
    logger.info("RL Risk Management Agent Training")
    logger.info("="*60)
    
    # Load episodes
    episodes = load_episodes(TRAINING_DATA_DIR)
    
    if len(episodes) == 0:
        logger.error("No episodes found! Please run prepare_rl_training_data.py first.")
        return
    
    # Split into train/val
    train_episodes, val_episodes = split_episodes(episodes, TRAIN_SPLIT)
    
    if len(train_episodes) == 0 or len(val_episodes) == 0:
        logger.error("Not enough episodes for training/validation split!")
        return
    
    # Print statistics
    logger.info(f"\nTraining Statistics:")
    logger.info(f"  Total episodes: {len(episodes)}")
    logger.info(f"  Training episodes: {len(train_episodes)}")
    logger.info(f"  Validation episodes: {len(val_episodes)}")
    
    train_returns = [ep['trade_return'] for ep in train_episodes]
    logger.info(f"  Training avg return: {np.mean(train_returns):.4f} ({np.mean(train_returns)*100:.2f}%)")
    logger.info(f"  Training win rate: {np.mean([r > 0 for r in train_returns])*100:.1f}%")
    
    val_returns = [ep['trade_return'] for ep in val_episodes]
    logger.info(f"  Validation avg return: {np.mean(val_returns):.4f} ({np.mean(val_returns)*100:.2f}%)")
    logger.info(f"  Validation win rate: {np.mean([r > 0 for r in val_returns])*100:.1f}%")
    
    # Train model
    model = train_ppo_agent(
        train_episodes,
        val_episodes,
        MODEL_SAVE_DIR,
        TENSORBOARD_LOG_DIR,
        total_timesteps=TOTAL_TIMESTEPS,
        n_envs=4  # Number of parallel environments
    )
    
    logger.info("\nTraining pipeline complete!")
    logger.info(f"To view TensorBoard: tensorboard --logdir {TENSORBOARD_LOG_DIR}")


if __name__ == "__main__":
    main()
