"""Sequential online trainer with durable episode and agent checkpoints."""

from __future__ import annotations

import json
import os
import signal
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from spark.agents import Agent, create_agent
from spark.environments import create_environment
from spark.models import (
    Episode,
    EpisodeStep,
    outcomes_from_payoffs,
    utc_now,
    winner_from_payoffs,
)
from spark.storage import RunDatabase


@dataclass
class TrainerConfig:
    run_id: str
    environment: str
    seed: int = 0
    run_directory: str = "runs"
    checkpoint_interval: int = 100
    episode_limit: int | None = None
    shared_agents: bool = True
    agent_memory_size: int = 100


class Trainer:
    def __init__(self, config: TrainerConfig) -> None:
        self.config = config
        self.run_path = Path(config.run_directory) / config.run_id
        self.run_path.mkdir(parents=True, exist_ok=True)
        self.database = RunDatabase(self.run_path / "run.sqlite")
        self.environment = create_environment(config.environment)
        self.agents: dict[int, Agent] = self._create_agents()
        self.agent_version = 0
        self.next_episode_number = 0
        self._stop = False
        self._checkpoint_requested = False
        self.checkpoint_path = self.run_path / "checkpoint.json"

        self.database.create_run(
            config.run_id,
            config.environment,
            self.agents[0].name,
            config.seed,
            asdict(config),
        )
        self._install_signal_handlers()

    def _create_agents(self) -> dict[int, Agent]:
        agent_config = {"memory_size": self.config.agent_memory_size}
        if self.config.shared_agents:
            agent = create_agent(
                self.config.environment,
                seed=self.config.seed,
                config=agent_config,
            )
            return {seat: agent for seat in range(self.environment.num_players)}
        return {
            seat: create_agent(
                self.config.environment,
                seed=self.config.seed + seat,
                config=agent_config,
            )
            for seat in range(self.environment.num_players)
        }

    def run(self, *, resume: bool = True) -> int:
        if resume and self.checkpoint_path.exists():
            self.load_checkpoint()
        self.database.update_run(status="running")
        self.database.add_event("started", "Training worker started.")

        try:
            while not self._should_stop():
                episode = self.run_episode(self.next_episode_number)
                # Durably store the experience before exposing it to update().
                self.database.save_episode(episode)
                unique_agents = self._unique_agents()
                metrics: dict[str, float] = {}
                for index, agent in enumerate(unique_agents):
                    agent_metrics = agent.update(episode)
                    prefix = f"agent_{index}." if len(unique_agents) > 1 else ""
                    metrics.update(
                        {f"{prefix}{name}": value for name, value in agent_metrics.items()}
                    )
                    agent.on_episode_end(episode)
                self.database.save_metrics(self.next_episode_number, metrics)
                self.agent_version += 1
                self.next_episode_number += 1
                self.database.update_run(
                    agent_version=self.agent_version,
                    episode_count=self.next_episode_number,
                )

                if (
                    self._checkpoint_requested
                    or self.next_episode_number % self.config.checkpoint_interval == 0
                ):
                    self.save_checkpoint()
                    self._checkpoint_requested = False
        except Exception as error:
            self.save_checkpoint()
            self.database.update_run(status="failed")
            self.database.add_event("failed", repr(error))
            raise

        self.save_checkpoint()
        final_status = (
            "completed"
            if self.config.episode_limit is not None
            and self.next_episode_number >= self.config.episode_limit
            else "stopped"
        )
        self.database.update_run(status=final_status)
        self.database.add_event(final_status, f"Worker ended at episode {self.next_episode_number}.")
        return self.next_episode_number

    def run_episode(self, episode_number: int) -> Episode:
        seed = self.config.seed + episode_number
        started = time.perf_counter()
        episode = Episode(
            run_id=self.config.run_id,
            episode_number=episode_number,
            environment=self.config.environment,
            seed=seed,
            agent_version=self.agent_version,
        )
        for agent in self._unique_agents():
            agent.on_episode_start(episode_number)

        decision = self.environment.reset(seed)
        while decision is not None:
            # Only this value is passed to the agent. Hidden state is logged separately.
            visible_before = decision.observation
            hidden_before = self.environment.hidden_state()
            agent = self.agents[decision.player_id]
            action = agent.select_action(
                visible_before,
                list(decision.legal_actions),
                decision.player_id,
            )
            if action not in decision.legal_actions:
                raise ValueError(
                    f"Agent selected illegal action {action}; legal={decision.legal_actions}"
                )
            player_id = decision.player_id
            legal_actions = list(decision.legal_actions)
            next_decision = self.environment.step(action)
            episode.steps.append(
                EpisodeStep(
                    index=len(episode.steps),
                    player_id=player_id,
                    agent_version=self.agent_version,
                    action=action,
                    action_name=self.environment.action_name(action),
                    legal_actions=legal_actions,
                    visible_before=visible_before,
                    visible_after=next_decision.observation if next_decision else {},
                    hidden_before=hidden_before,
                    hidden_after=self.environment.hidden_state(),
                )
            )
            decision = next_decision

        episode.payoffs = self.environment.payoffs()
        episode.outcomes = outcomes_from_payoffs(episode.payoffs)
        episode.winner = winner_from_payoffs(episode.payoffs)
        episode.completed_at = utc_now()
        episode.duration_seconds = time.perf_counter() - started
        return episode

    def save_checkpoint(self) -> Path:
        unique = self._unique_agents()
        payload = {
            "schema_version": 2,
            "config": asdict(self.config),
            "next_episode_number": self.next_episode_number,
            "agent_version": self.agent_version,
            "shared_agents": self.config.shared_agents,
            "agents": [agent.state_dict() for agent in unique],
        }
        temporary = self.checkpoint_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, self.checkpoint_path)
        self.database.save_checkpoint_record(
            self.agent_version,
            str(self.checkpoint_path),
            {"next_episode_number": self.next_episode_number},
        )
        return self.checkpoint_path

    def load_checkpoint(self) -> None:
        payload = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
        self.next_episode_number = int(payload["next_episode_number"])
        self.agent_version = int(
            payload.get("agent_version", payload.get("policy_version", 0))
        )
        states = list(payload.get("agents", payload.get("policies", [])))
        unique = self._unique_agents()
        if len(states) != len(unique):
            raise ValueError("Checkpoint agent topology does not match trainer config.")
        for agent, state in zip(unique, states):
            agent.load_state_dict(state)
        self.database.add_event(
            "resumed", f"Loaded checkpoint at episode {self.next_episode_number}."
        )

    def _unique_agents(self) -> list[Agent]:
        seen: set[int] = set()
        unique: list[Agent] = []
        for agent in self.agents.values():
            identity = id(agent)
            if identity not in seen:
                seen.add(identity)
                unique.append(agent)
        return unique

    def _should_stop(self) -> bool:
        if self._stop or self.database.stop_requested():
            return True
        limit = self.config.episode_limit
        return limit is not None and self.next_episode_number >= limit

    def _install_signal_handlers(self) -> None:
        def request_stop(_signum, _frame) -> None:
            self._stop = True
            self._checkpoint_requested = True

        def request_checkpoint(_signum, _frame) -> None:
            self._checkpoint_requested = True

        for name in ("SIGTERM", "SIGINT"):
            if hasattr(signal, name):
                signal.signal(getattr(signal, name), request_stop)
        if hasattr(signal, "SIGUSR1"):
            signal.signal(signal.SIGUSR1, request_checkpoint)
