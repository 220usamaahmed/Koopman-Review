import typing as _t
import minari
import gymnasium as gym
from pathlib import Path
from os import makedirs
from collections import deque
import numpy as np
import pickle


class Buffer:
    observations: _t.List
    actions: _t.List


class MujocoDataset:

    def __init__(
        self,
        env_name: str,
        tag: str | None = None,
        *,
        window_size: int = 50,
        random: bool = False,
        save_loc: str | None = None,
    ):
        self._env_name = env_name
        self._tag = tag
        self._save_loc = Path(
            save_loc
            if save_loc is not None
            else f"dataset-{env_name}-{tag if tag else 'uniform'}-{'random' if random else 'minari'}"
        )
        self._random = random
        self._window_size = window_size

        self._load_dataset()

    def __len__(self):
        return self._obs.shape[0]

    def __getitem__(self, idx):
        return (
            self._obs[idx, :, :],
            self._acts[idx, :, :],
        )

    def _load_dataset(self):
        print(f"Loading dataset from {self._save_loc}")
        obs_path = self._save_loc / "obs.npy"
        acts_path = self._save_loc / "acts.npy"
        if obs_path.exists() and acts_path.exists():
            print("Found npy data files, loading...")
            self._obs = np.load(obs_path)
            self._acts = np.load(acts_path)
        else:
            print("Npy data files not found, building dataset...")
            self._build_dataset()

    def _initialize_arrays(self):
        if self._random:
            env = gym.make(self._env_name, terminate_when_unhealthy=False)
            obs_len = env.observation_space.shape[0]
            acts_len = env.action_space.shape[0]
        else:
            dataset = minari.load_dataset(
                f"mujoco/{self._env_name}/{self._tag}", download=True
            )
            episode = next(dataset.iterate_episodes())
            obs_len = len(episode.observations[0])
            acts_len = len(episode.actions[0])

        self._max_trajs = int(1e5)
        self._obs = np.zeros((self._max_trajs, self._window_size, obs_len))
        self._acts = np.zeros((self._max_trajs, self._window_size, acts_len))
        self._traj_added = 0

    def _build_dataset(self):
        self._initialize_arrays()

        print(f"Building dataset for mujoco/{self._env_name}/{self._tag}")
        if self._random:
            self._build_random_dataset()
        else:
            self._build_minari_dataset()

    def _build_minari_dataset(self):
        dataset = minari.load_dataset(
            f"mujoco/{self._env_name}/{self._tag}", download=True
        )

        total_timesteps = 2e6
        current_timestep = 0

        for episode in dataset.iterate_episodes():
            current_timestep += len(episode.observations)

            print("Adding data from episode...")
            if self._add_from_episode(episode):
                break

            if current_timestep > total_timesteps:
                break

        self._save_dataset()
        print("Dataset build and saved.")

    def _build_random_dataset(self):
        print(f"Building random dataset for mujoco/{self._env_name}/{self._tag}")
        env = gym.make(self._env_name, terminate_when_unhealthy=False)
        while True:
            obs = env.reset()[0]
            done = False
            observations = []
            actions = []
            while not done:
                action = env.action_space.sample()
                observations.append(obs)
                actions.append(action)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            episode = Buffer()
            episode.observations = observations
            episode.actions = actions
            if self._add_from_episode(episode):
                break
        env.close()
        self._save_dataset()

    def _add_from_episode(self, episode: minari.EpisodeData | Buffer):
        de_size = self._window_size
        obs_size = len(episode.observations[0])
        acts_size = len(episode.actions[0])

        print(
            f"Processing episode: obs_size={obs_size}, acts_size={acts_size}, de_size={de_size}"
        )
        obs_de = deque([np.zeros(obs_size)] * de_size, maxlen=de_size)
        acts_de = deque([np.zeros(acts_size)] * de_size, maxlen=de_size)

        for i, (o, a) in enumerate(zip(episode.observations, episode.actions)):
            obs_de.append(o)
            acts_de.append(a)

            if i < de_size // 2 - 1:
                continue

            obs_list = list(obs_de)
            acts_list = list(acts_de)

            self._obs[self._traj_added, :, :] = np.array(obs_list)
            self._acts[self._traj_added, :, :] = np.array(acts_list)
            self._traj_added += 1
            if self._traj_added >= self._max_trajs:
                return True

        print(f"Added {self._obs.shape[0]} trajectories")

    def _save_dataset(self):
        if self._traj_added < self._max_trajs:
            self._obs = self._obs[: self._traj_added, :, :]
            self._acts = self._acts[: self._traj_added, :, :]

        print(f"Saving dataset to {self._save_loc}")
        makedirs(self._save_loc, exist_ok=True)

        np.save(self._save_loc / "obs.npy", self._obs)
        np.save(self._save_loc / "acts.npy", self._acts)
        print("Dataset saved and config.yaml updated.")


if __name__ == "__main__":
    print("Running MujocoDataset as main...")
    # d = MujocoDataset("ant", "expert-v0")
    d = MujocoDataset("HalfCheetah-v5", random=True)

    obs, act = d[1]

    print(len(d))
    print(obs.shape, obs.dtype)
    print(act.shape, act.dtype)

    print(obs[:25])

    print(d._obs.shape, d._acts.shape)
