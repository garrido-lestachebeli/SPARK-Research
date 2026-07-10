"""Interactive ASCII replay viewer for Blackjack episode reports."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
from gymnasium import Env

from main import EpisodeReport, run_episode
from storage import DEFAULT_EPISODES_FILE, append_episode, load_episodes, save_episodes

CARD_RANKS = {1: "A", 10: "10"}


def card_rank(value: int) -> str:
    return CARD_RANKS.get(value, str(value))


def draw_card(value: int, hidden: bool = False) -> list[str]:
    if hidden:
        return [
            "+-----+",
            "|#####|",
            "|#####|",
            "|#####|",
            "+-----+",
        ]
    rank = card_rank(value)
    padding = " " if len(rank) == 1 else ""
    return [
        "+-----+",
        f"| {rank}{padding} |",
        "|  *  |",
        f"| {rank}{padding} |",
        "+-----+",
    ]


def usable_ace_label(usable: bool) -> str:
    return "yes (soft hand)" if usable else "no"


class EpisodeReplayer:
    """Replay episode reports step-by-step with ASCII table visualization."""

    def __init__(
        self,
        reports: Optional[list[EpisodeReport] | EpisodeReport] = None,
        env: Optional[Env] = None,
        episode_file: str | Path = DEFAULT_EPISODES_FILE,
    ) -> None:
        self.env = env
        self.episode_file = Path(episode_file)
        self.reports: list[EpisodeReport] = []
        self.report_index = 0
        self.step_index = 0

        if reports is not None:
            if isinstance(reports, EpisodeReport):
                self.add_report(reports, select=True)
            else:
                self.set_reports(reports)
        elif self.episode_file.exists():
            self.load_from_file(self.episode_file)

    @property
    def current_report(self) -> Optional[EpisodeReport]:
        if not self.reports:
            return None
        return self.reports[self.report_index]

    @property
    def max_step_index(self) -> int:
        report = self.current_report
        if report is None:
            return 0
        return report.num_steps

    def set_reports(self, reports: list[EpisodeReport], *, report_index: int = 0) -> None:
        self.reports = list(reports)
        if self.reports:
            self.report_index = min(max(report_index, 0), len(self.reports) - 1)
        else:
            self.report_index = 0
        self.step_index = 0

    def add_report(self, report: EpisodeReport, *, select: bool = True) -> int:
        self.reports.append(report)
        index = len(self.reports) - 1
        if select:
            self.report_index = index
            self.step_index = 0
        return index

    def save_to_file(self, path: str | Path | None = None) -> Path:
        file_path = save_episodes(self.reports, path or self.episode_file)
        self.episode_file = file_path
        return file_path

    def load_from_file(
        self,
        path: str | Path | None = None,
        *,
        merge: bool = False,
    ) -> int:
        file_path = Path(path or self.episode_file)
        loaded = load_episodes(file_path)
        self.episode_file = file_path

        if merge:
            start_index = len(self.reports)
            self.reports.extend(loaded)
            self.report_index = start_index if loaded else self.report_index
        else:
            self.set_reports(loaded)

        self.step_index = min(self.step_index, self.max_step_index)
        return len(loaded)

    def load_new_report(self, seed: Optional[int] = None, *, persist: bool = True) -> EpisodeReport:
        if self.env is None:
            self.env = gym.make("Blackjack-v1")
        report = run_episode(self.env, seed=seed)
        self.add_report(report)
        if persist:
            append_episode(report, self.episode_file)
        return report

    def select_report(self, index: int) -> bool:
        if index < 0 or index >= len(self.reports):
            return False
        self.report_index = index
        self.step_index = min(self.step_index, self.max_step_index)
        return True

    def next_report(self) -> bool:
        if not self.reports:
            return False
        if self.report_index >= len(self.reports) - 1:
            return False
        self.report_index += 1
        self.step_index = min(self.step_index, self.max_step_index)
        return True

    def prev_report(self) -> bool:
        if not self.reports:
            return False
        if self.report_index <= 0:
            return False
        self.report_index -= 1
        self.step_index = min(self.step_index, self.max_step_index)
        return True

    def go_to_step(self, step_index: int) -> bool:
        if step_index < 0 or step_index > self.max_step_index:
            return False
        self.step_index = step_index
        return True

    def next_step(self) -> bool:
        if self.step_index >= self.max_step_index:
            return False
        self.step_index += 1
        return True

    def prev_step(self) -> bool:
        if self.step_index <= 0:
            return False
        self.step_index -= 1
        return True

    def first_step(self) -> None:
        self.step_index = 0

    def last_step(self) -> None:
        self.step_index = self.max_step_index

    def _current_state(self) -> dict[str, Any]:
        report = self.current_report
        if report is None:
            return {"player_sum": 0, "dealer_showing": 0, "usable_ace": False}

        if self.step_index == 0:
            return report.initial_state
        return report.steps[self.step_index - 1].state_after

    def _current_action(self) -> Optional[dict[str, Any]]:
        report = self.current_report
        if report is None or self.step_index == 0:
            return None
        step = report.steps[self.step_index - 1]
        return {
            "step": step.step,
            "action": step.action,
            "action_name": step.action_name,
            "reward": step.reward,
            "terminated": step.terminated,
            "truncated": step.truncated,
        }

    def _reward_so_far(self) -> float:
        report = self.current_report
        if report is None or self.step_index == 0:
            return 0.0
        return sum(step.reward for step in report.steps[: self.step_index])

    def _status_label(self) -> str:
        report = self.current_report
        if report is None:
            return "no report loaded"

        if self.step_index < report.num_steps:
            return "in progress"
        return report.outcome

    def render(self) -> str:
        report = self.current_report
        if report is None:
            return self._render_empty()

        state = self._current_state()
        action = self._current_action()
        dealer_card = int(state["dealer_showing"])
        player_sum = int(state["player_sum"])
        usable_ace = bool(state["usable_ace"])

        dealer_lines = draw_card(dealer_card)
        hidden_card = draw_card(0, hidden=True)
        table_width = 47

        lines: list[str] = [
            "+" + "=" * table_width + "+",
            "|" + "BLACKJACK - AGENT REPLAY".center(table_width) + "|",
            "+" + "=" * table_width + "+",
            "|  DEALER".ljust(table_width) + "|",
        ]

        for dealer_row, hidden_row in zip(dealer_lines, hidden_card):
            row = f"  {dealer_row}   {hidden_row}"
            lines.append("|" + row.ljust(table_width) + "|")

        lines.extend(
            [
                "|" + " " * table_width + "|",
                "|  PLAYER".ljust(table_width) + "|",
                "|" + f"  Hand total: {player_sum}".ljust(table_width) + "|",
                "|" + f"  Usable ace: {usable_ace_label(usable_ace)}".ljust(table_width) + "|",
                "|" + " " * table_width + "|",
            ]
        )

        if action is None:
            action_line = "  Waiting for agent's first action..."
        else:
            marker = "HIT" if action["action_name"] == "hit" else "STICK"
            action_line = (
                f"  >>> AGENT PLAYS: {marker} "
                f"(step {action['step']}, reward {action['reward']}) <<<"
            )
        lines.append("|" + action_line.ljust(table_width) + "|")

        lines.extend(
            [
                "+" + "=" * table_width + "+",
                "|"
                + (
                    f"  Report {self.report_index + 1}/{len(self.reports)}"
                    f"  |  Timeline {self.step_index}/{self.max_step_index}"
                ).ljust(table_width)
                + "|",
                "|"
                + (
                    f"  File: {self.episode_file.name}"
                    f"  |  Seed: {report.seed}"
                    f"  |  Reward: {self._reward_so_far()}"
                ).ljust(table_width)
                + "|",
                "|"
                + (
                    f"  Status: {self._status_label()}"
                ).ljust(table_width)
                + "|",
            ]
        )

        if self.step_index == report.num_steps:
            lines.append(
                "|"
                + (
                    f"  FINAL - total reward: {report.total_reward}"
                    f"  |  outcome: {report.outcome.upper()}"
                ).ljust(table_width)
                + "|"
            )

        lines.append("+" + "=" * table_width + "+")
        lines.append("")
        lines.append(self._controls_help())
        return "\n".join(lines)

    @staticmethod
    def _controls_help() -> str:
        return (
            "Controls: [n]ext step  [p]rev step  [f]irst  [l]ast  "
            "[>] next report  [<] prev report  [N]ew episode  "
            "[g #] go to step  [r #] go to report  [a]uto-play  "
            "[s]ave  [load path]  [q]uit"
        )

    @staticmethod
    def _render_empty() -> str:
        return (
            "No episode reports loaded.\n\n"
            "Press [N] to run a new episode, [load] to open a file, or [q] to quit."
        )

    def _clear_screen(self) -> None:
        if os.name == "nt":
            os.system("cls")
        else:
            os.system("clear")

    def display(self) -> None:
        self._clear_screen()
        print(self.render())

    def _parse_command(self, raw: str) -> tuple[str, list[str], str]:
        parts = raw.strip().split()
        if not parts:
            return "", [], raw.strip()
        return parts[0].lower(), parts[1:], parts[0]

    def _handle_command(self, command: str, args: list[str], token: str) -> bool:
        if command in {"q", "quit", "exit"}:
            return False

        if token == "N":
            seed = int(args[0]) if args else None
            report = self.load_new_report(seed=seed)
            print(f"Loaded new episode (report {self.report_index + 1}, seed={report.seed}).")
            input("Press Enter to continue...")
            return True

        if command in {"n", "next"}:
            if not self.next_step():
                print("Already at last step.")
                input("Press Enter to continue...")
            return True

        if command in {"p", "prev", "back"}:
            if not self.prev_step():
                print("Already at first step.")
                input("Press Enter to continue...")
            return True

        if command in {"f", "first"}:
            self.first_step()
            return True

        if command in {"l", "last"}:
            self.last_step()
            return True

        if command in {">", "nr", "next-report"}:
            if not self.next_report():
                print("Already at newest report.")
                input("Press Enter to continue...")
            return True

        if command in {"<", "pr", "prev-report"}:
            if not self.prev_report():
                print("Already at oldest report.")
                input("Press Enter to continue...")
            return True

        if command == "new":
            seed = int(args[0]) if args else None
            report = self.load_new_report(seed=seed)
            print(f"Loaded new episode (report {self.report_index + 1}, seed={report.seed}).")
            input("Press Enter to continue...")
            return True

        if command == "g" and args:
            if not self.go_to_step(int(args[0])):
                print(f"Step must be between 0 and {self.max_step_index}.")
                input("Press Enter to continue...")
            return True

        if command == "r" and args:
            if not self.select_report(int(args[0]) - 1):
                print(f"Report must be between 1 and {len(self.reports)}.")
                input("Press Enter to continue...")
            return True

        if command in {"a", "auto", "play"}:
            delay = float(args[0]) if args else 0.8
            self._auto_play(delay)
            return True

        if command == "s" or command == "save":
            path = Path(args[0]) if args else self.episode_file
            saved_path = self.save_to_file(path)
            print(f"Saved {len(self.reports)} episode(s) to {saved_path}.")
            input("Press Enter to continue...")
            return True

        if command == "load":
            path = Path(args[0]) if args else self.episode_file
            merge = bool(args[1:]) and args[1].lower() == "merge"
            try:
                count = self.load_from_file(path, merge=merge)
            except FileNotFoundError:
                print(f"Episode file not found: {path}")
                input("Press Enter to continue...")
                return True
            action = "Merged" if merge else "Loaded"
            print(f"{action} {count} episode(s) from {path}.")
            input("Press Enter to continue...")
            return True

        if command in {"h", "help", "?"}:
            print(self._controls_help())
            input("Press Enter to continue...")
            return True

        print(f"Unknown command: {token!r}. Press [h] for help.")
        input("Press Enter to continue...")
        return True

    def _auto_play(self, delay: float) -> None:
        while self.step_index < self.max_step_index:
            self.next_step()
            self.display()
            time.sleep(delay)
        input("Auto-play finished. Press Enter to continue...")

    def run(self) -> None:
        """Start the interactive replay loop."""
        if not self.reports:
            self.load_new_report()

        running = True
        while running:
            self.display()
            try:
                raw = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not raw:
                continue

            command, args, token = self._parse_command(raw)
            running = self._handle_command(command, args, token)

        print("Replay session ended.")


def main() -> None:
    episode_file = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_EPISODES_FILE
    replayer = EpisodeReplayer(episode_file=episode_file)

    if not replayer.reports:
        print(f"No episodes found at {episode_file}.")
        print("Starting with a new random episode (it will be saved automatically).")
        replayer.load_new_report()

    replayer.run()


if __name__ == "__main__":
    main()
