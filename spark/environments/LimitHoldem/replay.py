"""Interactive ASCII replay viewer for RLCard poker episode reports."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

from main import EpisodeReport, make_env, run_episode
from storage import (
    DEFAULT_EPISODES_FILE,
    append_episode,
    load_episodes,
    load_storage_metadata,
    save_episodes,
)


# ---------------------------------------------------------
# Card rendering
# ---------------------------------------------------------

def draw_card(card: Optional[str], hidden: bool = False) -> list[str]:
    """Render one poker card."""
    if hidden or card is None:
        return [
            "+-----+",
            "|#####|",
            "|#####|",
            "|#####|",
            "+-----+",
        ]

    # RLCard format:
    # HJ = hearts jack
    # SA = spades ace

    suit = card[0]
    rank = card[1:]

    padding = " " if len(rank) == 1 else ""

    return [
        "+-----+",
        f"| {rank}{padding}|",
        f"|  {suit}  |",
        f"| {rank}{padding}|",
        "+-----+",
    ]

def draw_cards(
    cards: list[str],
    hidden: bool = False,
) -> list[str]:
    """Render multiple cards side-by-side."""

    if not cards:
        return ["(no cards)"]

    blocks = [
        draw_card(card, hidden=hidden)
        for card in cards
    ]

    rows = []

    for row in range(len(blocks[0])):
        rows.append(
            "  ".join(
                block[row]
                for block in blocks
            )
        )

    return rows

# ---------------------------------------------------------
# Replay helpers
# ---------------------------------------------------------

def normalize_cards(value: Any) -> list[str]:
    """
    Convert RLCard card fields into a consistent list.

    Supports:
      "SA"
      ["SA", "HK"]
      None
    """

    if value is None:
        return []

    if isinstance(value, str):
        return [value]

    if isinstance(value, list):
        return [
            str(card)
            for card in value
        ]

    return []


def get_public_cards(info: dict[str, Any]) -> list[str]:
    """
    Supports:
      Leduc:
          public_card

      Limit Hold'em:
          public_cards
    """

    cards = normalize_cards(
        info.get("public_cards")
    )

    if cards:
        return cards

    return normalize_cards(
        info.get("public_card")
    )


def get_hands(info: dict[str, Any]) -> list[list[str]]:
    """
    Normalize hands.

    RLCard perfect information normally returns:

    [
       ["SA", "HK"],
       ["DQ", "C9"]
    ]

    """

    hands = info.get("hand_cards")

    if not hands:
        return []

    return [
        normalize_cards(hand)
        for hand in hands
    ]


def get_pot(info: dict[str, Any]) -> float:
    if info.get("pot") is not None:
        return float(info["pot"])

    chips = info.get("chips", [])

    try:
        return float(sum(chips))
    except Exception:
        return 0.0


class EpisodeReplayer:
    """Replay Limit Hold'em episode reports step-by-step with ASCII visualization."""



    def __init__(
            self,
            reports: Optional[list[EpisodeReport] | EpisodeReport] = None,
            episode_file: str | Path = DEFAULT_EPISODES_FILE,
    ) -> None:

        self.episode_file = Path(episode_file)

        self.reports: list[EpisodeReport] = []
        self.report_index = 0
        self.step_index = 0

        self.game_name = "unknown"

        if reports is not None:

            if isinstance(reports, EpisodeReport):
                self.add_report(
                    reports,
                    select=True
                )
            else:
                self.set_reports(reports)

        elif self.episode_file.exists():

            self.load_from_file(
                self.episode_file
            )

    @property
    def current_report(self):

        if not self.reports:
            return None

        return self.reports[
            self.report_index
        ]

    @property
    def max_step_index(self):

        report = self.current_report

        if report is None:
            return 0

        return report.num_steps

    def load_from_file(
            self,
            path=None,
            *,
            merge=False,
    ):

        file_path = Path(
            path or self.episode_file
        )

        self.game_name = load_storage_metadata(
            file_path
        ).get(
                "game",
                "unknown"
        )

        loaded = load_episodes(
            file_path
        )

        self.episode_file = file_path

        if merge:

            self.reports.extend(
                loaded
            )

        else:

            self.set_reports(
                loaded
            )

        return len(loaded)

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

    def save_to_file(
            self,
            path: str | Path | None = None
    ) -> Path:

        file_path = save_episodes(
            self.reports,
            path or self.episode_file,
        )

        self.episode_file = file_path

        return file_path


    def load_new_report(self, seed: Optional[int] = None, *, persist: bool = True) -> EpisodeReport:
        # RLCard seeds at env creation, so always make a fresh env for a seeded run.
        report = run_episode(make_env(seed=seed), seed=seed)
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
        if not self.reports or self.report_index >= len(self.reports) - 1:
            return False
        self.report_index += 1
        self.step_index = min(self.step_index, self.max_step_index)
        return True

    def prev_report(self) -> bool:
        if not self.reports or self.report_index <= 0:
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

    def _perfect_info_at_cursor(self) -> dict[str, Any]:
        report = self.current_report
        if report is None:
            return {}
        if self.step_index == 0:
            return report.initial_perfect_info or {}
        if self.step_index >= report.num_steps:
            return report.final_perfect_info or report.steps[-1].perfect_info
        return report.steps[self.step_index - 1].perfect_info or {}

    def _current_action(self) -> Optional[dict[str, Any]]:
        report = self.current_report
        if report is None or self.step_index == 0:
            return None
        step = report.steps[self.step_index - 1]
        return {
            "step": step.step,
            "player_id": step.player_id,
            "action": step.action,
            "action_name": step.action_name,
            "legal_actions": step.legal_actions,
        }

    def _status_label(self) -> str:
        report = self.current_report
        if report is None:
            return "no report loaded"
        if self.step_index < report.num_steps:
            return "in progress"
        if report.winner is None:
            return "draw" if all(outcome == "draw" for outcome in report.outcomes) else "finished"
        return f"player {report.winner} wins"

    def _show_all_hands(self) -> bool:
        report = self.current_report
        return report is not None and self.step_index >= report.num_steps

    def render(self) -> str:

        report = self.current_report

        if report is None:
            return "No episode loaded."

        info = self._perfect_info_at_cursor()

        action = self._current_action()

        hands = get_hands(info)

        #public_cards = get_public_cards(info)
        if self.step_index == 0:
            public_cards = report.initial_state.get("public_cards", [])
        elif self.step_index >= report.num_steps:
            public_cards = report.steps[-1].state_before.get("public_cards", [])
        else:
            public_cards = report.steps[self.step_index - 1].state_after.get("public_cards", [])
        chips = info.get(
            "chips",
            [0, 0]
        )

        current_player = info.get(
            "current_player"
        )

        show_all = (
                self.step_index >= report.num_steps
        )

        width = 60

        lines = [
            "+" + "=" * width + "+",
            "|" +
            f"{self.game_name.upper()} REPLAY".center(width)
            + "|",
            "+" + "=" * width + "+",
            "",
        ]

        # Players

        for player_id, hand in enumerate(hands):

            hidden = (
                    not show_all
                    and player_id != 0
            )

            lines.append(
                f"PLAYER {player_id}"
            )

            for row in draw_cards(
                    hand,
                    hidden
            ):
                lines.append(
                    row
                )

            if player_id < len(chips):
                lines.append(
                    f"chips: {chips[player_id]}"
                )

            lines.append("")

        # Board

        lines.append("BOARD")

        for row in draw_cards(
                public_cards
        ):
            lines.append(row)

        lines.append(
            f"Pot: {get_pot(info)}"
        )

        if current_player is not None:
            lines.append(
                f"To act: player {current_player}"
            )

        lines.append("")

        if action:

            lines.append(
                f">>> PLAYER {action['player_id']} "
                f"{action['action_name'].upper()} "
                f"(step {action['step']})"
            )

        else:

            lines.append(
                "Waiting for first action..."
            )

        lines.append("")

        lines.append(
            f"Timeline {self.step_index}/"
            f"{report.num_steps}"
        )

        if self.step_index == report.num_steps:
            lines.append(
                f"Final payoff: {report.payoffs}"
            )

            lines.append(
                f"Winner: {report.winner}"
            )

        lines.append("")

        lines.append(
            self._controls_help()
        )

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

        if command in {"s", "save"}:
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
