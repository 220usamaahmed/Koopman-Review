import minari
from torch.utils.data import Dataset
from pathlib import Path
from os import makedirs
from collections import deque
import numpy as np
import pickle
import yaml


class MujocoDataset(Dataset):

    def __init__(
        self,
        env_name: str,
        tag: str,
        *,
        input_traj_len: int = 25,
        output_traj_len: int = 25,
        save_loc: str | None = None,
    ):
        print(
            f"Initializing MujocoDataset with env_name={env_name}, tag={tag}, input_traj_len={input_traj_len}, output_traj_len={output_traj_len}, save_loc={save_loc}"
        )
        self._env_name = env_name
        self._tag = tag
        self._input_traj_len = input_traj_len
        self._output_traj_len = output_traj_len
        self._max_batch_size = input_traj_len + output_traj_len
        self._save_loc = Path(
            save_loc if save_loc is not None else f"dataset-{env_name}-{tag}"
        )

        self.X_obs = []
        self.X_acts = []
        self.Y_obs = []

        self._load_dataset()

    def __len__(self):
        return len(self.X_obs)

    def __getitem__(self, idx):
        return (
            self.X_obs[idx],
            self.X_acts[idx],
            self.Y_obs[idx],
        )

    def _load_dataset(self):
        print(f"Loading dataset from {self._save_loc}")
        if (self._save_loc / "config.yaml").exists():
            print("Found config.yaml, checking configuration...")
            with open(self._save_loc / "config.yaml", "r") as f:
                config = yaml.safe_load(f)
            if (
                config.get("input_traj_len") != self._input_traj_len
                or config.get("output_traj_len") != self._output_traj_len
                or config.get("tag") != self._tag
                or config.get("env_name") != self._env_name
            ):
                print("Config mismatch, rebuilding dataset...")
                self._build_dataset()
                return

        if (
            (self._save_loc / "X_obs.pkl").exists()
            and (self._save_loc / "X_acts.pkl").exists()
            and (self._save_loc / "Y_obs.pkl").exists()
        ):
            print("Found pickled data files, loading...")
            with open(self._save_loc / "X_obs.pkl", "rb") as f:
                self.X_obs = pickle.load(f)
            with open(self._save_loc / "X_acts.pkl", "rb") as f:
                self.X_acts = pickle.load(f)
            with open(self._save_loc / "Y_obs.pkl", "rb") as f:
                self.Y_obs = pickle.load(f)
            print(
                f"Loaded {len(self.X_obs)} X_obs, {len(self.X_acts)} X_acts, {len(self.Y_obs)} Y_obs"
            )
        else:
            print("Pickled data files not found, building dataset...")
            self._build_dataset()

    def _build_dataset(self):
        print(f"Building dataset for mujoco/{self._env_name}/{self._tag}")
        dataset = minari.load_dataset(
            f"mujoco/{self._env_name}/{self._tag}", download=True
        )

        for episode in dataset.iterate_episodes():
            print("Adding data from episode...")
            self._add_from_episode(episode)

        self._save_dataset()
        print("Dataset build and saved.")

    def _add_from_episode(self, episode: minari.EpisodeData):
        de_size = self._input_traj_len + self._output_traj_len
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

            if i < self._input_traj_len:
                continue

            obs_list = list(obs_de)
            acts_list = list(acts_de)

            self.X_obs.append(np.array(obs_list[: self._input_traj_len]))
            self.X_acts.append(np.array(acts_list[: self._input_traj_len]))
            self.Y_obs.append(np.array(obs_list[self._input_traj_len :]))
        print(
            f"Added {len(self.X_obs)} X_obs, {len(self.X_acts)} X_acts, {len(self.Y_obs)} Y_obs so far."
        )

    def _save_dataset(self):
        print(f"Saving dataset to {self._save_loc}")
        makedirs(self._save_loc, exist_ok=True)

        with open(self._save_loc / "X_obs.pkl", "wb") as f:
            pickle.dump(self.X_obs, f)
        with open(self._save_loc / "X_acts.pkl", "wb") as f:
            pickle.dump(self.X_acts, f)
        with open(self._save_loc / "Y_obs.pkl", "wb") as f:
            pickle.dump(self.Y_obs, f)
        config = {
            "input_traj_len": self._input_traj_len,
            "output_traj_len": self._output_traj_len,
            "tag": self._tag,
            "env_name": self._env_name,
        }
        with open(self._save_loc / "config.yaml", "w") as f:
            yaml.safe_dump(config, f)
        print("Dataset saved and config.yaml updated.")


if __name__ == "__main__":
    print("Running MujocoDataset as main...")
    d = MujocoDataset("halfcheetah", "expert-v0")

    X_obs, X_act, Y_obs = d[0]

    print(X_obs.shape)
    print(X_act.shape)
    print(Y_obs.shape)
