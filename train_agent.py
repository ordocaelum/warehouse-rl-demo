import argparse
import functools
import importlib.metadata
import json
import logging
import platform
import threading
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.logger import configure as configure_sb3_logger
from warehouse_env import WarehouseEnv

log = logging.getLogger("warehouse")

# Every training run writes its outputs to its own folder under here.
RUNS_DIR = "runs"
# Steps between checkpoints (model + replay buffer) in <run>/checkpoints/.
CHECKPOINT_FREQ = 10_000


class RewardCallback(BaseCallback):
    """Track episode rewards during training, with optional live visualizer support."""

    def __init__(self, visualizer=None, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.current_reward = 0
        self._visualizer = visualizer
        self._episode_num = 0
        self._step_in_ep = 0
        self._stepping_episode = False  # S pressed: run to the episode's end

    def _on_step(self) -> bool:
        reward = self.locals.get("rewards", [0])[0]
        done = bool(self.locals.get("dones", [False])[0])
        self.current_reward += reward
        self._step_in_ep += 1

        # Push live update to visualizer (non-blocking)
        if self._visualizer is not None:
            avg_reward = None
            window = self.episode_rewards[-50:]
            if len(self.episode_rewards) >= 50:
                avg_reward = float(np.mean(window))

            epsilon = 1.0
            if hasattr(self.model, "exploration_rate"):
                epsilon = float(self.model.exploration_rate)

            self._visualizer.put_update({
                "episode":   self._episode_num,
                "step":      self._step_in_ep,
                "reward":    float(reward),
                "ep_reward": float(self.current_reward),
                "avg_reward": avg_reward,
                "epsilon":   epsilon,
                # Forwarded as the env reported it; the env itself is never
                # read. On an episode's last step this is still that step,
                # although the VecEnv has already reset the env.
                "render_state": self.locals["infos"][0].get("render_state"),
            })

            # Honour pause/resume from visualizer; bail out if it stops running.
            # S while paused runs to the end of the current episode, then
            # pauses again on that episode's final frame.
            if self._stepping_episode and done:
                self._stepping_episode = False
            if not self._stepping_episode:
                while self._visualizer.is_paused() and self._visualizer.is_running():
                    if self._visualizer.consume_step_request():
                        self._stepping_episode = True
                        break
                    time.sleep(0.05)

        # End of episode bookkeeping
        if done:
            self.episode_rewards.append(self.current_reward)
            if self._visualizer is not None:
                self._visualizer.put_episode_reward(self.current_reward)
            self.current_reward = 0
            self._episode_num += 1
            self._step_in_ep = 0

        return True


# DQN settings that have no CLI flag or dashboard slider. Both entry points
# build their model through make_env() and make_dqn(), so the same visible
# settings always train the same way.
LEARNING_STARTS = 500
TARGET_UPDATE_INTERVAL = 500
EXPLORATION_FRACTION = 0.2


def new_run_dir(runs_dir=RUNS_DIR, grid_size: int = 8, num_packages: int = 1,
                seed: int | None = None) -> Path:
    """Create and return an empty folder for one training run's outputs."""
    base = Path(runs_dir) / f"{time.strftime('%Y%m%d-%H%M%S')}_g{grid_size}_p{num_packages}_s{seed}"
    path, n = base, 1
    while path.exists():
        path = base.with_name(f"{base.name}-{n}")
        n += 1
    path.mkdir(parents=True)
    return path


def write_run_config(run_dir, **config) -> None:
    """Record a run's settings, and the library versions it ran with, in config.json."""
    versions = {
        name: importlib.metadata.version(name)
        for name in ("gymnasium", "stable-baselines3", "torch", "numpy")
    }
    data = {**config, "python": platform.python_version(), "versions": versions}
    (Path(run_dir) / "config.json").write_text(json.dumps(data, indent=2))


def attach_run_logger(model, run_dir, stdout: bool = False) -> None:
    """Send SB3's training metrics to run_dir: progress.csv, and TensorBoard if installed."""
    formats = ["csv"]
    try:
        import tensorboard  # noqa: F401
        formats.append("tensorboard")
    except ImportError:
        log.warning("tensorboard is not installed; logging metrics to CSV only")
    if stdout:
        formats.append("stdout")
    model.set_logger(configure_sb3_logger(str(run_dir), formats))


def replay_buffer_path(model_path) -> Path:
    """Where the replay buffer saved with ``model_path`` (with or without .zip) lives."""
    model_path = Path(model_path)
    if model_path.suffix == ".zip":
        model_path = model_path.with_suffix("")
    return model_path.with_name(model_path.name + "_replay_buffer.pkl")


def save_run_model(model, model_path) -> None:
    """Save the model and its replay buffer, so training can resume exactly."""
    model.save(model_path)
    model.save_replay_buffer(replay_buffer_path(model_path))


def load_run_model(model_path, env, seed: int | None = None) -> DQN:
    """Load a saved model to continue training it on ``env``.

    Restores everything SB3 saves (weights, optimizer state, hyperparameters,
    step count) plus the replay buffer if it was saved alongside. Raises
    ValueError if ``env``'s observation space differs from the model's.
    """
    model = DQN.load(model_path, env=env)
    if seed is not None:
        model.set_random_seed(seed)
    buffer_path = replay_buffer_path(model_path)
    if buffer_path.exists():
        model.load_replay_buffer(buffer_path)
    return model


def read_run_config(run_dir) -> dict:
    """The config.json a run wrote, or {} if there is none."""
    path = Path(run_dir) / "config.json"
    return json.loads(path.read_text()) if path.exists() else {}


def checkpoint_callback(run_dir, save_freq: int = CHECKPOINT_FREQ) -> CheckpointCallback:
    """Save model + replay buffer to run_dir/checkpoints every ``save_freq`` steps."""
    return CheckpointCallback(
        save_freq=save_freq,
        save_path=str(Path(run_dir) / "checkpoints"),
        name_prefix="model",
        save_replay_buffer=True,
    )


def find_latest_model(runs_dir=RUNS_DIR) -> Path | None:
    """The most recently written runs/*/model.zip, or None."""
    models = sorted(Path(runs_dir).glob("*/model.zip"), key=lambda p: p.stat().st_mtime)
    return models[-1] if models else None


def make_env(grid_size: int = 8, num_packages: int = 1, max_steps: int = 300,
             step_cost: float = 0.05, seed: int | None = None, monitor_dir=None):
    """The single-env DummyVecEnv (with Monitor) that both entry points train on.

    With ``monitor_dir``, Monitor writes per-episode reward/length/time to
    ``monitor_dir/0.monitor.csv``.
    """
    make_fn = functools.partial(
        WarehouseEnv,
        grid_size=grid_size,
        num_packages=num_packages,
        max_steps=max_steps,
        step_cost=step_cost,
    )
    return make_vec_env(
        make_fn, n_envs=1, seed=seed,
        monitor_dir=str(monitor_dir) if monitor_dir is not None else None,
    )


def make_dqn(env, learning_rate: float = 1e-3, buffer_size: int = 10000,
             exploration_initial: float = 1.0, exploration_final: float = 0.1,
             exploration_fraction: float = EXPLORATION_FRACTION,
             seed: int | None = None, verbose: int = 0) -> DQN:
    """The DQN configuration shared by the CLI and the dashboard."""
    # Note: DQN is based on Q-learning principles
    return DQN(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=LEARNING_STARTS,
        target_update_interval=TARGET_UPDATE_INTERVAL,
        exploration_fraction=exploration_fraction,
        exploration_initial_eps=exploration_initial,
        exploration_final_eps=exploration_final,
        seed=seed,
        verbose=verbose,
    )


def train_agent(
    total_timesteps: int = 100000,
    visualizer=None,
    grid_size: int = 8,
    num_packages: int = 1,
    max_steps: int = 300,
    step_cost: float = 0.05,
    learning_rate: float = 1e-3,
    buffer_size: int = 10000,
    exploration_initial: float = 1.0,
    exploration_final: float = 0.1,
    exploration_fraction: float = EXPLORATION_FRACTION,
    seed: int | None = None,
    run_dir=None,
    checkpoint_freq: int = CHECKPOINT_FREQ,
):
    """Train DQN agent on warehouse environment.

    ``seed`` seeds the environment, the action space, numpy, Python's
    ``random`` and torch (via SB3).  ``None`` means an unseeded run.

    Outputs go to ``run_dir`` (a new folder under ``runs/`` if not given):
    config.json, model.zip and its replay buffer, checkpoints/ (every
    ``checkpoint_freq`` steps; 0 disables), 0.monitor.csv, progress.csv
    and TensorBoard event files.
    """
    env_config = dict(grid_size=grid_size, num_packages=num_packages,
                      max_steps=max_steps, step_cost=step_cost)
    agent_config = dict(learning_rate=learning_rate, buffer_size=buffer_size,
                        exploration_initial=exploration_initial,
                        exploration_final=exploration_final,
                        exploration_fraction=exploration_fraction)
    if run_dir is None:
        run_dir = new_run_dir(grid_size=grid_size, num_packages=num_packages, seed=seed)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_run_config(
        run_dir, env=env_config,
        agent={**agent_config, "learning_starts": LEARNING_STARTS,
               "target_update_interval": TARGET_UPDATE_INTERVAL},
        seed=seed, total_timesteps=total_timesteps,
    )
    log.info("Run directory: %s", run_dir)

    env = make_env(**env_config, seed=seed, monitor_dir=run_dir)
    model = make_dqn(env, **agent_config, seed=seed, verbose=1)
    attach_run_logger(model, run_dir, stdout=True)

    # Train with callback
    callback = RewardCallback(visualizer=visualizer)
    callbacks = [callback]
    if checkpoint_freq > 0:
        callbacks.append(checkpoint_callback(run_dir, checkpoint_freq))
    model.learn(total_timesteps=total_timesteps, callback=CallbackList(callbacks))

    save_run_model(model, run_dir / "model")
    log.info("Model saved to %s", run_dir / "model.zip")

    return model, callback, env

def visualize_training(callback: RewardCallback, path="training_progress.png",
                       show: bool = True):
    """Plot training progress to ``path`` (and show it unless ``show`` is False)."""
    import matplotlib.pyplot as plt  # only needed here; keeps the dashboard light

    plt.figure(figsize=(12, 5))
    
    # Episode rewards
    plt.subplot(1, 2, 1)
    plt.plot(callback.episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Progress: Episode Rewards")
    plt.grid(True)
    
    # Moving average
    plt.subplot(1, 2, 2)
    window = 50
    if len(callback.episode_rewards) > window:
        moving_avg = np.convolve(callback.episode_rewards, np.ones(window)/window, mode='valid')
        plt.plot(moving_avg)
        plt.xlabel("Episode")
        plt.ylabel("Average Reward (50-ep window)")
        plt.title("Training Progress: Moving Average")
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(path)
    if show:
        plt.show()
    plt.close()

def test_agent(model_path, num_episodes: int = 3,
               render: bool = True, seed: int | None = None, **env_kwargs):
    """Run the trained agent greedily and return each episode's total reward.

    ``env_kwargs`` (grid_size, num_packages, max_steps, step_cost) must match
    the configuration the model was trained on.
    """
    # "human" prints the grid after every step
    env = WarehouseEnv(render_mode="human" if render else None, **env_kwargs)
    model = DQN.load(model_path)
    if model.observation_space != env.observation_space:
        env.close()
        raise ValueError(
            f"{model_path} was trained on a different configuration "
            f"(observation space {model.observation_space}, this env has "
            f"{env.observation_space}); pass the training grid_size and num_packages"
        )

    episode_rewards = []
    for episode in range(num_episodes):
        obs, _ = env.reset(seed=seed if episode == 0 else None)
        done = False
        total_reward = 0
        
        print(f"\n--- Episode {episode + 1} ---")
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        
        print(f"Episode {episode + 1} Total Reward: {total_reward}")
        episode_rewards.append(total_reward)
    env.close()
    return episode_rewards

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train warehouse DQN agent")
    parser.add_argument(
        "--visualize", action="store_true",
        help="Open a live PyGame visualizer while training"
    )
    parser.add_argument(
        "--timesteps", type=int, default=100000,
        help="Total training timesteps (default: 100000)"
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed for env, agent and torch (default: 0)"
    )
    parser.add_argument("--grid-size", type=int, default=8, help="Grid size (default: 8)")
    parser.add_argument("--packages", type=int, default=1, help="Packages (default: 1)")
    parser.add_argument("--max-steps", type=int, default=300, help="Episode step limit (default: 300)")
    parser.add_argument("--step-cost", type=float, default=0.05, help="Per-step cost (default: 0.05)")
    parser.add_argument(
        "--checkpoint-freq", type=int, default=CHECKPOINT_FREQ,
        help=f"Save a checkpoint every N steps, 0 to disable (default: {CHECKPOINT_FREQ})",
    )
    parser.add_argument(
        "--runs-dir", default=RUNS_DIR,
        help=f"Folder for run outputs; each run gets a subfolder (default: {RUNS_DIR})",
    )
    args = parser.parse_args()

    run_dir = new_run_dir(args.runs_dir, args.grid_size, args.packages, args.seed)
    logging.basicConfig(
        level=logging.INFO, format="%(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(run_dir / "train.log")],
    )

    # The same environment configuration is used for training and testing.
    env_config = dict(
        grid_size=args.grid_size,
        num_packages=args.packages,
        max_steps=args.max_steps,
        step_cost=args.step_cost,
    )

    def train(visualizer=None):
        return train_agent(
            total_timesteps=args.timesteps,
            visualizer=visualizer,
            seed=args.seed,
            run_dir=run_dir,
            checkpoint_freq=args.checkpoint_freq,
            **env_config,
        )

    log.info("Training agent...")
    if args.visualize:
        from visualizer_pygame import WarehouseVisualizer
        visualizer = WarehouseVisualizer(grid_size=args.grid_size)
        result = {}

        def train_in_background():
            try:
                result["value"] = train(visualizer)
            except BaseException as exc:  # re-raised on the main thread
                result["error"] = exc
            finally:
                visualizer.stop()  # training finished: close the window

        # pygame needs the main thread (macOS), so training runs beside it.
        # Closing the window early lets training continue without display.
        trainer = threading.Thread(target=train_in_background, daemon=True)
        trainer.start()
        log.info("Live visualizer started. Controls: SPACE=pause  S=step  ↑↓=zoom  drag=pan  R=reset")
        visualizer.run()
        trainer.join()
        if "error" in result:
            raise result["error"]
        model, callback, env = result["value"]
    else:
        model, callback, env = train()

    env.close()

    # Visualize
    log.info("Plotting training progress...")
    visualize_training(callback, run_dir / "training_progress.png")

    # Test
    log.info("Testing trained agent...")
    test_agent(run_dir / "model", num_episodes=3, seed=args.seed, **env_config)
    log.info("Outputs are in %s", run_dir)
