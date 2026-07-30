"""Deep Q-learning agent for Leduc Hold'em."""

from __future__ import annotations

import random
from collections import deque
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from spark.agents.base import Agent
from spark.models import Episode


# ============================================================
# Neural network
# ============================================================


class QNetwork(nn.Module):
    """Simple MLP approximator for Leduc observations."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# Replay buffer
# ============================================================


class ReplayBuffer:
    def __init__(
        self,
        capacity: int = 100_000,
    ) -> None:

        self.buffer = deque(
            maxlen=capacity
        )


    def add(
        self,
        state: list[float],
        action: int,
        reward: float,
        next_state: list[float] | None,
        next_legal: list[int],
        done: bool,
    ) -> None:

        self.buffer.append(
            (
                state,
                action,
                reward,
                next_state,
                next_legal,
                done,
            )
        )


    def sample(
        self,
        batch_size: int,
    ):

        return random.sample(
            self.buffer,
            batch_size,
        )


    def __len__(self) -> int:
        return len(self.buffer)


    def state_dict(self) -> list[Any]:
        return list(self.buffer)


    def load_state_dict(
        self,
        state: list[Any],
    ) -> None:

        self.buffer.clear()
        self.buffer.extend(state)


# ============================================================
# DQN Agent
# ============================================================


class LeducDQNAgentBet(Agent):

    agent_id = "leduc-dqn-bet"
    name = "Leduc Deep Q Network Betting History"
    environment_id = "leduc-holdem"

    action_names = {
        0: "call",
        1: "raise",
        2: "fold",
        3: "check",
    }


    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        seed: int = 0,
    ) -> None:

        super().__init__(
            config,
            seed=seed,
        )


        torch.manual_seed(
            self.seed
        )


        # ----------------------------
        # Hyperparameters
        # ----------------------------

        self.gamma = float(
            self.config.get(
                "gamma",
                0.99,
            )
        )

        self.lr = float(
            self.config.get(
                "lr",
                1e-3,
            )
        )

        self.batch_size = int(
            self.config.get(
                "batch_size",
                4,
            )
        )


        self.epsilon = float(
            self.config.get(
                "epsilon",
                1.0,
            )
        )

        self.epsilon_min = float(
            self.config.get(
                "epsilon_min",
                0.05,
            )
        )

        self.epsilon_decay = float(
            self.config.get(
                "epsilon_decay",
                0.999,
            )
        )


        self.target_update = int(
            self.config.get(
                "target_update",
                1000,
            )
        )


        self.replay_capacity = int(
            self.config.get(
                "replay_size",
                100_000,
            )
        )


        # ----------------------------
        # Runtime state
        # ----------------------------

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )


        self.state_dim: int | None = None

        self.q_network: QNetwork | None = None
        self.target_network: QNetwork | None = None
        self.optimizer: optim.Optimizer | None = None


        self.loss_function = nn.SmoothL1Loss()


        self.memory = ReplayBuffer(
            self.replay_capacity
        )


        self.gradient_steps = 0



    # ========================================================
    # Encoding
    # ========================================================



    @staticmethod
    def encode(observation):
        MAX_HISTORY = 4
        # RLCard encoded observation
        state = list(observation.get("encoded", []))

        # Flatten betting history
        history = []

        ACTION_TO_ID = {
            "call": 0,
            "raise": 1,
            "fold": 2,
            "check": 3,
        }

        for player, action in observation.get("action_record", []):

            player_vector = [0.0, 0.0]
            player_vector[player] = 1.0
            history.extend(player_vector)

            # Convert string action to integer
            if isinstance(action, str):
                action = ACTION_TO_ID[action]

            action_vector = [0.0] * 4
            action_vector[action] = 1.0

            history.extend(action_vector)

        # Keep only the last MAX_HISTORY actions
        FEATURES_PER_ACTION = 6
        history = history[-FEATURES_PER_ACTION * MAX_HISTORY:]
        #history = history[-2 * MAX_HISTORY:]

        # Pad with -1 if there are fewer than MAX_HISTORY actions
        while len(history) < FEATURES_PER_ACTION * MAX_HISTORY:
            history.append(-1.0)

        state.extend(history)

        return np.asarray(state, dtype=np.float32)


    # ========================================================
    # Network initialization
    # ========================================================


    def _build_networks(
        self,
        state_dim: int,
    ) -> None:

        self.state_dim = state_dim


        self.q_network = QNetwork(
            state_dim,
            len(self.action_names),
        ).to(self.device)


        self.target_network = QNetwork(
            state_dim,
            len(self.action_names),
        ).to(self.device)


        self.target_network.load_state_dict(
            self.q_network.state_dict()
        )


        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=self.lr,
        )



    # ========================================================
    # Action selection
    # ========================================================


    def select_action(
        self,
        observation: dict[str, Any],
        legal_actions: list[int],
        player_id: int,
    ) -> int:

        del player_id


        if not legal_actions:
            raise ValueError(
                "No legal actions."
            )


        state = self.encode(
            observation
        )


        if self.q_network is None:
            self._build_networks(
                len(state)
            )


        if (
            self.training
            and self._random.random() < self.epsilon
        ):
            return self._random.choice(
                legal_actions
            )


        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        )


        with torch.no_grad():

            values = self.q_network(
                state_tensor
            )


        masked = torch.full_like(
            values,
            -1e9,
        )


        for action in legal_actions:
            masked[action] = 0


        return int(
            torch.argmax(
                values + masked
            ).item()
        )


    # ========================================================
    # Episode learning
    # ========================================================


    def update(
        self,
        episode: Episode,
    ) -> dict[str, float]:


        steps = [
            step
            for step in episode.steps
            if step.player_id == self.seat_id
        ]


        for index, step in enumerate(steps):

            next_step = (
                steps[index + 1]
                if index + 1 < len(steps)
                else None
            )


            state = self.encode(
                step.visible_before
            )


            if self.q_network is None:
                self._build_networks(
                    len(state)
                )


            if next_step is None:

                next_state = None
                next_legal = []
                done = True

                reward = float(
                    episode.payoffs[
                        self.seat_id
                    ]
                )

            else:

                next_state = self.encode(
                    next_step.visible_before
                )

                next_legal = (
                    next_step.legal_actions
                )

                done = False

                reward = 0.0


            self.memory.add(
                state,
                step.action,
                reward,
                next_state,
                next_legal,
                done,
            )


        metrics = self.train_step()


        super().update(
            episode
        )


        return metrics



    # ========================================================
    # Gradient update
    # ========================================================


    def train_step(
        self,
    ) -> dict[str, float]:

        if (
            self.q_network is None
            or len(self.memory) < self.batch_size
        ):
            return {
                "loss": 0.0,
                "epsilon": self.epsilon,
                "replay_size": float(len(self.memory)),
            }


        assert self.optimizer is not None
        assert self.target_network is not None


        batch = self.memory.sample(
            self.batch_size
        )


        losses = []


        for (
            state,
            action,
            reward,
            next_state,
            next_legal,
            done,
        ) in batch:


            state_tensor = torch.tensor(
                state,
                dtype=torch.float32,
                device=self.device,
            )


            current_q = self.q_network(
                state_tensor
            )[action]


            target = torch.tensor(
                reward,
                dtype=torch.float32,
                device=self.device,
            )


            if not done and next_state is not None:

                next_tensor = torch.tensor(
                    next_state,
                    dtype=torch.float32,
                    device=self.device,
                )


                with torch.no_grad():

                    # Double DQN:
                    with torch.no_grad():
                        q = self.q_network(next_tensor)

                        mask = torch.full_like(q, -1e9)

                        for a in next_legal:
                            mask[a] = 0

                        next_action = torch.argmax(q + mask).item()

                        target += (
                                self.gamma
                                * self.target_network(next_tensor)[next_action]
                        )


            losses.append(
                self.loss_function(
                    current_q,
                    target,
                )
            )


        loss = torch.stack(
            losses
        ).mean()


        self.optimizer.zero_grad()

        loss.backward()


        torch.nn.utils.clip_grad_norm_(
            self.q_network.parameters(),
            10,
        )


        self.optimizer.step()


        self.gradient_steps += 1


        if (
            self.gradient_steps
            %
            self.target_update
            == 0
        ):

            self.target_network.load_state_dict(
                self.q_network.state_dict()
            )


        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay,
        )


        return {
            "loss": float(loss.item()),
            "epsilon": self.epsilon,
            "replay_size": float(len(self.memory)),
        }



    # ========================================================
    # Checkpointing
    # ========================================================


    def state_dict(
        self,
    ) -> dict[str, Any]:

        state = super().state_dict()


        state.update(
            {
                "gamma": self.gamma,
                "lr": self.lr,
                "epsilon": self.epsilon,
                "state_dim": self.state_dim,
                "gradient_steps": self.gradient_steps,
                "memory": self.memory.state_dict(),
            }
        )

        if self.q_network is not None:
            state["q_network"] = {
                key: value.cpu().tolist()
                for key, value in self.q_network.state_dict().items()
            }

            state["target_network"] = {
                key: value.cpu().tolist()
                for key, value in self.target_network.state_dict().items()
            }


        return state



    def load_state_dict(
        self,
        state: dict[str, Any],
    ) -> None:

        super().load_state_dict(
            state
        )


        self.gamma = float(
            state.get(
                "gamma",
                self.gamma,
            )
        )


        self.lr = float(
            state.get(
                "lr",
                self.lr,
            )
        )


        self.epsilon = float(
            state.get(
                "epsilon",
                self.epsilon,
            )
        )


        self.gradient_steps = int(
            state.get(
                "gradient_steps",
                0,
            )
        )


        self.memory.load_state_dict(
            list(
                state.get(
                    "memory",
                    [],
                )
            )
        )


        state_dim = state.get(
            "state_dim"
        )


        if state_dim is not None:

            self._build_networks(
                int(state_dim)
            )

            if "q_network" in state:
                q_state = {
                    key: torch.tensor(value)
                    for key, value in state["q_network"].items()
                }

                target_state = {
                    key: torch.tensor(value)
                    for key, value in state["target_network"].items()
                }

                self.q_network.load_state_dict(
                    q_state
                )

                self.target_network.load_state_dict(
                    target_state
                )