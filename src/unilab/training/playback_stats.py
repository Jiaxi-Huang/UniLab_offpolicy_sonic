"""Low-overhead playback rollout statistics for NumPy environments."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np


class PlaybackStats:
    """Collect termination, episode, and reward summaries while stepping an env."""

    def __init__(self, env: Any, *, output_path: str | os.PathLike[str]) -> None:
        self.env, self.output_path = env, Path(output_path)
        self.steps = 0
        self.term_counts = {name: 0 for name in env.termination_manager.active_terms}
        self.terminated = self.truncated = 0
        self.episode_lengths: list[int] = []
        self.episode_returns: list[float] = []
        self._length = np.zeros(env.num_envs, dtype=np.int64)
        self._return = np.zeros(env.num_envs, dtype=np.float64)
        self._reward_values: list[float] = []

    def observe(self, state: Any) -> None:
        self.steps += 1
        reward = np.asarray(state.reward, dtype=np.float64).reshape(-1)
        self._length += 1
        self._return += reward
        self._reward_values.extend(float(x) for x in reward)
        tm = self.env.termination_manager
        for name in tm.active_terms:
            self.term_counts[name] += int(np.count_nonzero(tm.get_term(name)))
        terminated = np.asarray(state.terminated, dtype=bool).reshape(-1)
        truncated = np.asarray(state.truncated, dtype=bool).reshape(-1)
        self.terminated += int(np.count_nonzero(terminated))
        self.truncated += int(np.count_nonzero(truncated))
        for idx in np.flatnonzero(terminated | truncated):
            self.episode_lengths.append(int(self._length[idx]))
            self.episode_returns.append(float(self._return[idx]))
            self._length[idx] = 0
            self._return[idx] = 0.0

    @staticmethod
    def _summary(values: list[float] | list[int]) -> dict[str, float | int | None]:
        if not values:
            return {"count": 0, "mean": None, "min": None, "max": None}
        arr = np.asarray(values, dtype=np.float64)
        return {
            "count": len(values),
            "mean": float(arr.mean()),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }

    def write(self, *, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        dt = float(getattr(self.env.cfg, "ctrl_dt", 0.0))
        payload = {
            "metadata": metadata or {},
            "steps": self.steps,
            "envs": int(self.env.num_envs),
            "ctrl_dt": dt,
            "simulated_seconds": self.steps * dt,
            "terminations": {
                "by_reason": self.term_counts,
                "terminated": self.terminated,
                "truncated": self.truncated,
                "episodes_completed": len(self.episode_lengths),
            },
            "episodes": {
                "length": self._summary(self.episode_lengths),
                "return": self._summary(self.episode_returns),
            },
            "reward": self._summary(self._reward_values),
            "in_progress_envs": int(np.count_nonzero(self._length)),
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.output_path.with_suffix(self.output_path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(tmp, self.output_path)
        return payload
