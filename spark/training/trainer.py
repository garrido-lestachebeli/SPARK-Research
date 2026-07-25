"""Sequential online trainer with durable episode and agent checkpoints."""

from __future__ import annotations

import json
import os
import random
import signal
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from spark.agents import Agent, create_agent, default_agent_id
from spark.environments import create_environment
from spark.league import update_ratings
from spark.models import (
    Episode,
    EpisodeStep,
    outcomes_from_payoffs,
    utc_now,
    winner_from_payoffs,
)
from spark.storage import LeagueMemberStore, RunDatabase


@dataclass
class TrainerConfig:
    run_id: str
    environment: str
    table_size: int = 2
    seat_configs: list[dict[str, Any]] | None = None
    # Legacy fields retained so schema-v2 checkpoints and old CLI calls still load.
    agents: list[str] | None = None
    seed: int = 0
    run_directory: str = "data"
    checkpoint_interval: int = 100
    episode_limit: int | None = None
    shared_agents: bool | None = None
    agent_memory_size: int = 100
    update_interval: int = 1
    league_reroll_interval: int = 0


class Trainer:
    def __init__(self, config: TrainerConfig) -> None:
        self.config = config
        self.config.update_interval = max(1, int(config.update_interval))
        self.config.league_reroll_interval = max(0, int(config.league_reroll_interval))
        self.run_path = Path(config.run_directory) / config.run_id
        self.run_path.mkdir(parents=True, exist_ok=True)
        self.database = RunDatabase(self.run_path / "run.sqlite")
        self.environment = create_environment(config.environment, config.table_size)
        self.seat_configs = self._normalize_seat_configs()
        self.agents: dict[int, Agent] = self._create_agents()
        self.agent_version = 0
        self.next_episode_number = 0
        self._stop = False
        self._checkpoint_requested = False
        self.checkpoint_path = self.run_path / "checkpoint.json"
        self._episode_buffer: list[Episode] = []
        self.league = LeagueMemberStore(config.run_directory)
        self._reroll_rng = random.Random(config.seed + 17)

        self.database.create_run(
            config.run_id,
            config.environment,
            self.agents[0].name,
            config.seed,
            asdict(config),
        )
        self._install_signal_handlers()

    def _create_agents(self) -> dict[int, Agent]:
        agents: dict[int, Agent] = {}
        for seat, seat_config in enumerate(self.seat_configs):
            agent = create_agent(
                seat_config["agent_id"],
                seed=self.config.seed + seat,
                config={
                    "memory_size": self.config.agent_memory_size,
                    "training": seat_config["mode"] == "train",
                    "seat_id": seat,
                },
            )
            if agent.environment_id != self.config.environment:
                raise ValueError(
                    f"Agent {agent.agent_id!r} is for {agent.environment_id!r}, "
                    f"not {self.config.environment!r}."
                )
            agent.set_training(seat_config["mode"] == "train")
            agents[seat] = agent
        return agents

    def _normalize_seat_configs(self) -> list[dict[str, Any]]:
        values = self.config.seat_configs
        if values is None and self.config.agents:
            values = [
                {"source": "fresh", "agent_id": agent_id, "mode": "train"}
                for agent_id in self.config.agents
            ]
        if values is None:
            default_id = default_agent_id(self.config.environment)
            values = [
                {"source": "fresh", "agent_id": default_id, "mode": "train"}
                for _ in range(self.environment.num_players)
            ]
        if len(values) != self.environment.num_players:
            raise ValueError(
                f"Expected {self.environment.num_players} seat configurations, "
                f"received {len(values)}."
            )
        normalized: list[dict[str, Any]] = []
        for seat, value in enumerate(values):
            agent_id = str(value.get("agent_id", "")).strip()
            mode = str(value.get("mode", "train")).lower()
            source = str(value.get("source") or "fresh").lower()
            if not agent_id:
                raise ValueError(f"Seat {seat} has no agent_id.")
            if mode not in {"train", "evaluate"}:
                raise ValueError(f"Seat {seat} has invalid mode {mode!r}.")
            if source not in {"fresh", "league_member", "league_random"}:
                source = "fresh"
            normalized.append(
                {
                    "source": source,
                    "agent_id": agent_id,
                    "mode": mode,
                    "snapshot_id": value.get("snapshot_id"),
                    "league_member_id": value.get("league_member_id"),
                }
            )
        return normalized

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
                self._episode_buffer.append(episode)
                for agent in self.agents.values():
                    agent.on_episode_end(episode)

                metrics: dict[str, float] = {}
                if len(self._episode_buffer) >= self.config.update_interval:
                    metrics = self._flush_update_buffer()
                self.database.save_metrics(self.next_episode_number, metrics)

                self.next_episode_number += 1
                self.database.update_run(
                    agent_version=self.agent_version,
                    episode_count=self.next_episode_number,
                )

                if self._should_reroll():
                    self._reroll_league_random_seats()

                if (
                    self._checkpoint_requested
                    or self.next_episode_number % self.config.checkpoint_interval == 0
                ):
                    self._persist_league_training_states()
                    self.save_checkpoint()
                    self._checkpoint_requested = False
        except Exception as error:
            self._flush_update_buffer()
            self._persist_league_training_states()
            self.save_checkpoint()
            self.database.update_run(status="failed")
            self.database.add_event("failed", repr(error))
            raise

        self._flush_update_buffer()
        self._persist_league_training_states()
        self.save_checkpoint()
        final_status = (
            "completed"
            if self.config.episode_limit is not None
            and self.next_episode_number >= self.config.episode_limit
            else "stopped"
        )
        self.database.update_run(status=final_status)
        self.database.add_event(
            final_status, f"Worker ended at episode {self.next_episode_number}."
        )
        return self.next_episode_number

    def _flush_update_buffer(self) -> dict[str, float]:
        if not self._episode_buffer:
            return {}
        metrics: dict[str, float] = {}
        self._update_league_elo_from_buffer()
        updated = False
        for episode in self._episode_buffer:
            for seat, agent in self.agents.items():
                if not agent.training:
                    continue
                agent_metrics = agent.update(episode)
                updated = True
                metrics.update(
                    {
                        f"seat_{seat}.{name}": value
                        for name, value in agent_metrics.items()
                    }
                )
        self._episode_buffer.clear()
        if updated:
            self.agent_version += 1
            self._persist_league_training_states()
        return metrics

    def _league_member_ids_by_seat(self) -> dict[int, str]:
        mapping: dict[int, str] = {}
        for seat, seat_config in enumerate(self.seat_configs):
            member_id = seat_config.get("league_member_id")
            if member_id:
                mapping[seat] = str(member_id)
        return mapping

    def _update_league_elo_from_buffer(self) -> None:
        """Transfer ELO between League seats from win rates over the N-episode window."""
        member_by_seat = self._league_member_ids_by_seat()
        if len(member_by_seat) < 2:
            return
        episodes = self._episode_buffer
        if not episodes:
            return

        wins: dict[str, int] = {member_id: 0 for member_id in member_by_seat.values()}
        losses: dict[str, int] = {member_id: 0 for member_id in member_by_seat.values()}
        draws: dict[str, int] = {member_id: 0 for member_id in member_by_seat.values()}
        for episode in episodes:
            for seat, member_id in member_by_seat.items():
                outcome = "draw"
                if seat < len(episode.outcomes):
                    outcome = str(episode.outcomes[seat])
                if outcome == "win":
                    wins[member_id] += 1
                elif outcome == "loss":
                    losses[member_id] += 1
                else:
                    draws[member_id] += 1

        window = float(len(episodes))
        winrates = {
            member_id: wins[member_id] / window for member_id in member_by_seat.values()
        }
        ratings: dict[str, float] = {}
        for member_id in winrates:
            try:
                ratings[member_id] = float(self.league.get(member_id)["elo"])
            except Exception as error:
                self.database.add_event(
                    "league_elo_failed",
                    f"Could not load ELO for {member_id}: {error!r}",
                )
                return

        updated = update_ratings(ratings, winrates)
        for member_id, new_elo in updated.items():
            try:
                self.league.set_elo(member_id, new_elo)
                self.league.record_match_stats(
                    member_id,
                    wins=wins[member_id],
                    losses=losses[member_id],
                    draws=draws[member_id],
                )
            except Exception as error:
                self.database.add_event(
                    "league_elo_failed",
                    f"Could not save ELO for {member_id}: {error!r}",
                )
                continue
            self.database.add_event(
                "league_elo",
                (
                    f"{member_id}: {ratings[member_id]:.1f} -> {new_elo:.1f} "
                    f"(win_rate={winrates[member_id]:.3f} over {int(window)} episodes)"
                ),
            )

    def _should_reroll(self) -> bool:
        interval = self.config.league_reroll_interval
        if interval <= 0:
            return False
        return self.next_episode_number > 0 and self.next_episode_number % interval == 0

    def _reroll_league_random_seats(self) -> None:
        changed = False
        for seat, seat_config in enumerate(self.seat_configs):
            if seat_config.get("source") != "league_random":
                continue
            current_id = seat_config.get("league_member_id")
            agent = self.agents[seat]
            if agent.training and current_id:
                self.league.save_agent_state(str(current_id), agent.state_dict())
            member = self.league.sample_active(
                self.config.environment,
                rng=self._reroll_rng,
                exclude_id=str(current_id) if current_id else None,
            )
            member_id = str(member["id"])
            agent_id = str(member["agent_id"])
            mode = "train" if str(member.get("mode") or "").lower() == "train" else "evaluate"
            replacement = create_agent(
                agent_id,
                seed=self.config.seed + seat + self.next_episode_number,
                config={
                    "memory_size": self.config.agent_memory_size,
                    "training": mode == "train",
                    "seat_id": seat,
                },
            )
            replacement.load_state_dict(dict(member["state"]))
            replacement.seat_id = seat
            replacement.set_training(mode == "train")
            self.agents[seat] = replacement
            seat_config["agent_id"] = agent_id
            seat_config["mode"] = mode
            seat_config["league_member_id"] = member_id
            changed = True
            self.database.add_event(
                "league_reroll",
                f"Seat {seat} re-rolled to {member.get('name')} ({member_id}).",
            )
        if changed:
            self.save_checkpoint()

    def _persist_league_training_states(self) -> None:
        for seat, seat_config in enumerate(self.seat_configs):
            member_id = seat_config.get("league_member_id")
            if not member_id:
                continue
            agent = self.agents[seat]
            if not agent.training:
                continue
            try:
                self.league.save_agent_state(str(member_id), agent.state_dict())
            except Exception as error:
                self.database.add_event(
                    "league_persist_failed",
                    f"Seat {seat} member {member_id}: {error!r}",
                )

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
        for agent in self.agents.values():
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
        payload = {
            "schema_version": 3,
            "config": asdict(self.config),
            "next_episode_number": self.next_episode_number,
            "agent_version": self.agent_version,
            "seat_configs": self.seat_configs,
            "agents": [self.agents[seat].state_dict() for seat in sorted(self.agents)],
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
        saved_seats = list(payload.get("seat_configs") or [])
        if saved_seats and len(saved_seats) == len(self.seat_configs):
            self.seat_configs = self._normalize_seat_configs_from(saved_seats)
            # Rebuild agents if agent_ids changed due to prior league re-rolls.
            self.agents = self._create_agents()
        states = list(payload.get("agents", payload.get("policies", [])))
        ordered_agents = [self.agents[seat] for seat in sorted(self.agents)]
        if len(states) == 1 and len(ordered_agents) > 1:
            # Schema-v2 shared-agent checkpoints are cloned into each configured seat.
            states = states * len(ordered_agents)
        if len(states) != len(ordered_agents):
            raise ValueError("Checkpoint seat topology does not match trainer config.")
        for seat, (agent, state) in enumerate(zip(ordered_agents, states)):
            # Agent identity may have changed after a league re-roll; recreate if needed.
            expected_id = self.seat_configs[seat]["agent_id"]
            if agent.agent_id != expected_id:
                agent = create_agent(
                    expected_id,
                    seed=self.config.seed + seat,
                    config={
                        "memory_size": self.config.agent_memory_size,
                        "training": self.seat_configs[seat]["mode"] == "train",
                        "seat_id": seat,
                    },
                )
                self.agents[seat] = agent
            agent.load_state_dict(state)
            agent.seat_id = seat
            agent.set_training(self.seat_configs[seat]["mode"] == "train")
        saved_config = dict(payload.get("config") or {})
        if "update_interval" in saved_config:
            self.config.update_interval = max(1, int(saved_config["update_interval"]))
        if "league_reroll_interval" in saved_config:
            self.config.league_reroll_interval = max(
                0, int(saved_config["league_reroll_interval"])
            )
        self.database.add_event(
            "resumed", f"Loaded checkpoint at episode {self.next_episode_number}."
        )

    def _normalize_seat_configs_from(
        self, values: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        self.config.seat_configs = values
        return self._normalize_seat_configs()

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
