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
    CallbackList,
    BaseCallback
)
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import torch
import torch.nn as nn

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
TOTAL_TIMESTEPS = 5e6  # Total training steps (use early stopping)
LEARNING_RATE = 1e-4  # Learning rate (further reduced for stability with scaled rewards)
BATCH_SIZE = 128  # Batch size for stable training
N_STEPS = 2048  # Steps per update
N_EPOCHS = 5  # Optimization epochs per update (reduced to prevent overfitting)
GAMMA = 0.99  # Discount factor
GAE_LAMBDA = 0.95  # GAE lambda
ENT_COEF = 0.1  # Entropy coefficient (exploration) - increased to prevent premature convergence
VF_COEF = 1.0  # Value function coefficient (increased to stabilize value learning)
MAX_GRAD_NORM = 0.5  # Maximum gradient norm for clipping

# Model architecture configuration
POLICY_LAYERS = [512, 256, 256, 32]  # Policy network hidden layers
VALUE_LAYERS = [512, 256, 256, 32]   # Value network hidden layers
ACTIVATION_FN = 'tanh'  # Activation function: 'tanh', 'relu', or 'elu'
CHECKPOINT_FREQ = 10000  # Save checkpoint every N steps
EVAL_FREQ = 5000  # Evaluate every N steps
EVAL_EPISODES = 10  # Number of episodes for evaluation

# Early stopping configuration
ENABLE_EARLY_STOPPING = False  # Enable early stopping
EARLY_STOPPING_PATIENCE = 50  # Number of evaluations without improvement before stopping
EARLY_STOPPING_MIN_DELTA = 0.0  # Minimum change to qualify as improvement
EARLY_STOPPING_MONITOR = 'mean_reward'  # Metric to monitor: 'mean_reward', 'mean_ep_length', or 'loss'
EARLY_STOPPING_MODE = 'max'  # 'max' for reward/ep_length (higher is better), 'min' for loss (lower is better)

# Training/Validation split
TRAIN_SPLIT = 0.9  # 90% training, 10% validation


class DynamicEvalCallback(BaseCallback):
    """
    Custom evaluation callback that recreates environments with different episodes each evaluation.
    This ensures evaluation uses diverse episodes and reflects actual learning progress.
    """
    
    def __init__(
        self,
        val_episodes: List[Dict],
        eval_freq: int,
        n_eval_episodes: int,
        model_save_dir: Path,
        log_path: Path,
        deterministic: bool = True,
        verbose: int = 1
    ):
        super().__init__(verbose=verbose)
        self.val_episodes = val_episodes
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.model_save_dir = model_save_dir
        self.log_path = log_path
        self.deterministic = deterministic
        self.last_mean_reward = None
        self.last_mean_ep_length = None
        self.best_mean_reward = float('-inf')
        self.eval_count = 0
        
        log_path.mkdir(parents=True, exist_ok=True)
        
    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Recreate evaluation environments with random episodes
            eval_env_factory = create_eval_env_factory(self.val_episodes, seed=123 + self.eval_count)
            n_eval_envs = min(self.n_eval_episodes, len(self.val_episodes), 10)
            eval_env = make_vec_env(
                eval_env_factory,
                n_envs=n_eval_envs,
                vec_env_cls=DummyVecEnv
            )
            
            # Evaluate current model and get episode rewards/lengths
            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=self.deterministic,
                return_episode_rewards=True
            )
            
            # Normalize rewards by episode length to make them comparable to training
            # Training episodes are shorter, so we normalize to reward per step
            if episode_rewards and episode_lengths:
                # Calculate reward per step for each episode
                rewards_per_step = [r / max(l, 1) for r, l in zip(episode_rewards, episode_lengths)]
                mean_reward = np.mean(rewards_per_step)
                std_reward = np.std(rewards_per_step)
                # Also keep total reward for reference
                mean_total_reward = np.mean(episode_rewards)
                std_total_reward = np.std(episode_rewards)
            else:
                mean_reward = 0.0
                std_reward = 0.0
                mean_total_reward = 0.0
                std_total_reward = 0.0
            
            mean_ep_length = np.mean(episode_lengths) if episode_lengths else 0.0
            
            # Store normalized reward (per step) for early stopping comparison
            self.last_mean_reward = mean_reward
            self.last_mean_ep_length = mean_ep_length
            self.eval_count += 1
            
            # Log to TensorBoard (same format as EvalCallback)
            self.logger.record("eval/mean_reward", mean_reward)
            self.logger.record("eval/mean_ep_length", mean_ep_length)
            self.logger.record("eval/std_reward", std_reward)
            self.logger.record("eval/mean_total_reward", mean_total_reward)
            self.logger.record("eval/std_total_reward", std_total_reward)
            self.logger.dump(step=self.num_timesteps)
            
            # Log results (show both normalized and total for reference)
            if self.verbose >= 1:
                logger.info(f"Eval num_timesteps={self.num_timesteps}, "
                          f"episode_reward_per_step={mean_reward:.2f} +/- {std_reward:.2f} "
                          f"(total: {mean_total_reward:.2f} +/- {std_total_reward:.2f})")
                logger.info(f"Episode length: {mean_ep_length:.2f}")
            
            # Save best model
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                best_model_path = self.model_save_dir / "best_model.zip"
                self.model.save(str(best_model_path))
                if self.verbose >= 1:
                    logger.info(f"New best model saved with mean reward: {mean_reward:.2f}")
            
            # Clean up
            eval_env.close()
            
        return True


class EarlyStoppingCallback(BaseCallback):
    """
    Custom early stopping callback that monitors validation loss/metrics.
    Stops training if the monitored metric doesn't improve for a specified number of evaluations.
    """
    
    def __init__(
        self,
        eval_callback,  # Can be EvalCallback or DynamicEvalCallback
        patience: int = EARLY_STOPPING_PATIENCE,
        min_delta: float = EARLY_STOPPING_MIN_DELTA,
        monitor: str = EARLY_STOPPING_MONITOR,
        mode: str = EARLY_STOPPING_MODE,
        verbose: int = 1
    ):
        """
        Initialize early stopping callback.
        
        Args:
            eval_callback: EvalCallback or DynamicEvalCallback instance to monitor
            patience: Number of evaluations without improvement before stopping
            min_delta: Minimum change to qualify as improvement
            monitor: Metric to monitor ('mean_reward', 'mean_ep_length', or 'loss')
            mode: 'max' for reward/ep_length, 'min' for loss
            verbose: Verbosity level
        """
        super().__init__(verbose=verbose)
        self.eval_callback = eval_callback
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        
        # Track best value and patience counter
        self.best_value = float('-inf') if mode == 'max' else float('inf')
        self.patience_counter = 0
        self.wait = 0
        self.stopped_epoch = 0
        self.last_eval_step = -1
        self.last_mean_reward = None
        
        logger.info(f"Early stopping callback initialized:")
        logger.info(f"  Monitor: {monitor}")
        logger.info(f"  Mode: {mode}")
        logger.info(f"  Patience: {patience} evaluations")
        logger.info(f"  Min delta: {min_delta}")
    
    def _on_step(self) -> bool:
        """
        Called at each step. Check if early stopping should be triggered.
        
        Returns:
            True if training should continue, False if should stop
        """
        # Only check after evaluation has occurred
        # Check if a new evaluation has happened by comparing current mean_reward with last
        if not hasattr(self.eval_callback, 'last_mean_reward'):
            return True
        
        current_mean_reward = self.eval_callback.last_mean_reward
        
        # Check if this is a new evaluation (mean_reward changed)
        if self.last_mean_reward is not None and current_mean_reward == self.last_mean_reward:
            return True  # No new evaluation yet
        
        # New evaluation detected
        self.last_mean_reward = current_mean_reward
        
        # Get current metric value from evaluation callback
        if self.monitor == 'mean_reward':
            if not hasattr(self.eval_callback, 'last_mean_reward'):
                return True
            current_value = self.eval_callback.last_mean_reward
            if current_value is None:
                return True
            try:
                current_value = float(current_value)
                if np.isnan(current_value) or np.isinf(current_value):
                    return True  # Skip if NaN or Inf
            except (TypeError, ValueError):
                return True  # Skip if not numeric
        
        elif self.monitor == 'mean_ep_length':
            if not hasattr(self.eval_callback, 'last_mean_ep_length'):
                return True
            current_value = self.eval_callback.last_mean_ep_length
            if current_value is None:
                return True
            try:
                current_value = float(current_value)
                if np.isnan(current_value) or np.isinf(current_value):
                    return True  # Skip if NaN or Inf
            except (TypeError, ValueError):
                return True  # Skip if not numeric
        
        elif self.monitor == 'loss':
            # For loss, we'll use negative mean reward as proxy
            # (lower reward = higher loss, so we minimize negative reward)
            if not hasattr(self.eval_callback, 'last_mean_reward'):
                return True
            reward = self.eval_callback.last_mean_reward
            if reward is None:
                return True
            try:
                current_value = -float(reward)
                if np.isnan(current_value) or np.isinf(current_value):
                    return True  # Skip if NaN or Inf
            except (TypeError, ValueError):
                return True  # Skip if not numeric
            # Use min mode for loss
            effective_mode = 'min'
        else:
            logger.warning(f"Unknown monitor metric: {self.monitor}, using mean_reward")
            if not hasattr(self.eval_callback, 'last_mean_reward'):
                return True
            current_value = self.eval_callback.last_mean_reward
            if current_value is None:
                return True
            try:
                current_value = float(current_value)
                if np.isnan(current_value) or np.isinf(current_value):
                    return True
            except (TypeError, ValueError):
                return True
            effective_mode = self.mode
        
        # Use effective mode for loss, otherwise use configured mode
        if self.monitor == 'loss':
            check_mode = effective_mode
        else:
            check_mode = self.mode
        
        # Check for improvement
        if check_mode == 'max':
            improved = current_value >= self.best_value + self.min_delta
        else:  # mode == 'min'
            improved = current_value <= self.best_value - self.min_delta
        
        if improved:
            self.best_value = current_value
            self.wait = 0
            if self.verbose >= 1:
                logger.info(f"Early stopping: {self.monitor} improved to {current_value:.4f} (best: {self.best_value:.4f})")
        else:
            self.wait += 1
            if self.verbose >= 1:
                logger.info(f"Early stopping: {self.monitor} did not improve. Wait: {self.wait}/{self.patience} (current: {current_value:.4f}, best: {self.best_value:.4f})")
        
        # Check if patience exceeded
        if self.wait >= self.patience:
            self.stopped_epoch = self.num_timesteps
            logger.info(f"\n{'='*60}")
            logger.info(f"Early stopping triggered at {self.num_timesteps} timesteps!")
            logger.info(f"  Best {self.monitor}: {self.best_value:.4f}")
            logger.info(f"  Stopped after {self.patience} evaluations without improvement")
            logger.info(f"{'='*60}\n")
            return False  # Stop training
        
        return True  # Continue training


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
    Create evaluation environment factory that randomly samples episodes.
    
    Args:
        val_episodes: List of validation episodes
        seed: Random seed
    
    Returns:
        Function that returns an environment
    """
    # Store episodes list for random sampling
    episodes_list = val_episodes
    
    def _make_env(rank: int = 0):
        # Create a new random state for each environment creation
        # This ensures different episodes are selected
        local_rng = np.random.RandomState(seed + rank + int(np.random.random() * 1000000))
        # Randomly sample a validation episode
        episode_idx = local_rng.randint(0, len(episodes_list))
        episode = episodes_list[episode_idx]
        # Use different seed for each environment instance
        env_seed = seed + rank + local_rng.randint(0, 10000)
        env = make_env(episode, rank=rank, seed=env_seed)()
        # Store episode list in env for later random resets
        env._available_episodes = episodes_list
        env._episode_rng = np.random.RandomState(env_seed)
        return env
    
    return _make_env


def train_ppo_agent(
    train_episodes: List[Dict],
    val_episodes: List[Dict],
    model_save_dir: Path,
    tensorboard_log_dir: Path,
    total_timesteps: int = TOTAL_TIMESTEPS,
    n_envs: int = 32
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
    # Use multiple environments to evaluate different episodes in parallel
    eval_env_factory = create_eval_env_factory(val_episodes, seed=123)
    n_eval_envs = min(EVAL_EPISODES, len(val_episodes), 10)  # Use up to 10 parallel envs
    eval_env = make_vec_env(
        eval_env_factory,
        n_envs=n_eval_envs,
        vec_env_cls=DummyVecEnv
    )
    logger.info(f"Created {n_eval_envs} evaluation environments for {EVAL_EPISODES} episodes")
    
    # Create PPO model with more complex architecture
    logger.info("Creating PPO model with complex architecture...")
    
    # Map activation function string to PyTorch activation
    activation_map = {
        'tanh': torch.nn.Tanh,
        'relu': torch.nn.ReLU,
        'elu': torch.nn.ELU,
        'leaky_relu': torch.nn.LeakyReLU
    }
    activation_fn = activation_map.get(ACTIVATION_FN.lower(), torch.nn.Tanh)
    
    policy_kwargs = dict(
        # Deeper and wider network architecture
        # Separate networks for policy (pi) and value function (vf)
        # Policy network: Multiple hidden layers with decreasing width (bottleneck design)
        # Value network: Multiple hidden layers with decreasing width
        net_arch=[
            dict(pi=POLICY_LAYERS,  # Policy network layers
                 vf=VALUE_LAYERS)   # Value network layers
        ],
        # Use activation function
        activation_fn=activation_fn
    )
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("Using CPU (consider using GPU for faster training with complex architecture)")
    
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
        device=device
    )
    
    logger.info(f"Model architecture:")
    logger.info(f"  Policy network: {POLICY_LAYERS} ({len(POLICY_LAYERS)} hidden layers)")
    logger.info(f"  Value network: {VALUE_LAYERS} ({len(VALUE_LAYERS)} hidden layers)")
    logger.info(f"  Activation function: {ACTIVATION_FN}")
    # Estimate parameters: sum of layer sizes squared
    pi_params = sum(POLICY_LAYERS[i] * POLICY_LAYERS[i+1] for i in range(len(POLICY_LAYERS)-1))
    vf_params = sum(VALUE_LAYERS[i] * VALUE_LAYERS[i+1] for i in range(len(VALUE_LAYERS)-1))
    total_params = (pi_params + vf_params) + sum(POLICY_LAYERS) + sum(VALUE_LAYERS)  # Approximate
    logger.info(f"  Estimated parameters: ~{total_params // 1000}K")
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
    
    # Evaluation callback - use custom callback that recreates environments each time
    eval_callback = DynamicEvalCallback(
        val_episodes=val_episodes,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=EVAL_EPISODES,
        model_save_dir=model_save_dir,
        log_path=model_save_dir / "eval_logs",
        deterministic=True,
        verbose=1
    )
    callbacks.append(eval_callback)
    
    # Close the pre-created eval_env since we'll recreate it each time
    eval_env.close()
    
    # Early stopping callback
    if ENABLE_EARLY_STOPPING:
        early_stopping_callback = EarlyStoppingCallback(
            eval_callback=eval_callback,
            patience=EARLY_STOPPING_PATIENCE,
            min_delta=EARLY_STOPPING_MIN_DELTA,
            monitor=EARLY_STOPPING_MONITOR,
            mode=EARLY_STOPPING_MODE,
            verbose=1
        )
        callbacks.append(early_stopping_callback)
        logger.info(f"Early stopping enabled:")
        logger.info(f"  Monitor: {EARLY_STOPPING_MONITOR}")
        logger.info(f"  Mode: {EARLY_STOPPING_MODE}")
        logger.info(f"  Patience: {EARLY_STOPPING_PATIENCE} evaluations")
        logger.info(f"  Min delta: {EARLY_STOPPING_MIN_DELTA}")
    else:
        logger.info("Early stopping disabled")
    
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
    # Note: eval_env was already closed when we switched to DynamicEvalCallback
    # (it creates its own environments each evaluation)
    
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
    train_lengths = [ep['episode_length'] for ep in train_episodes]
    logger.info(f"  Training avg return: {np.mean(train_returns):.4f} ({np.mean(train_returns)*100:.2f}%)")
    logger.info(f"  Training win rate: {np.mean([r > 0 for r in train_returns])*100:.1f}%")
    logger.info(f"  Training avg episode length: {np.mean(train_lengths):.1f} periods")
    logger.info(f"  Training episode length range: {np.min(train_lengths)} - {np.max(train_lengths)} periods")
    
    val_returns = [ep['trade_return'] for ep in val_episodes]
    val_lengths = [ep['episode_length'] for ep in val_episodes]
    logger.info(f"  Validation avg return: {np.mean(val_returns):.4f} ({np.mean(val_returns)*100:.2f}%)")
    logger.info(f"  Validation win rate: {np.mean([r > 0 for r in val_returns])*100:.1f}%")
    logger.info(f"  Validation avg episode length: {np.mean(val_lengths):.1f} periods")
    logger.info(f"  Validation episode length range: {np.min(val_lengths)} - {np.max(val_lengths)} periods")
    
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
