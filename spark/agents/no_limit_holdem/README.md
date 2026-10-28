# No-Limit Hold'em for a DQN agent

This file is the field guide for the No-Limit Texas Hold'em environment.

`spark/agents/no_limit_holdem/agent.py` is shared by every No-Limit agent. It
holds the action names and the one-hot card encoder. It is not itself an agent,
and it does not hold learning settings.

`spark/agents/no_limit_holdem/dqn.py` is the DQN. Gamma, the learning rate,
epsilon, the replay buffer, and the belief bookkeeping live on
`NoLimitHoldemDQNAgent` in that file. The same file holds the four functions
that are still empty: `build_q_network`, `prior`, `bayes_update`, and
`belief_features`. Each one raises `NotImplementedError` until you write it.
Play continues without them: the agent stores transitions and picks a legal
action at random.

Train one hand at a time (`--episode-mode hand`). Use `gamma = 1`, which is the
default. Every decision in the hand is then trained on the full chip result.

## What one hand is

Heads-up or a larger table, from 2 to 10 players. The campaign form and the
training command set the buy-in and the blinds. Defaults are a starting stack
of 50 chips, a small blind of 1, and a big blind of 2, so a default hand is
25 big blinds deep. The big blind must be at least the small blind, and the
starting stack must cover the big blind. Chip features in the observation vector, and
the terminal reward, divide by the big blind of that hand.

RLCard does not accept an arbitrary raise size. Every decision is one of five
actions:

| Id | Action | What it does |
| --- | --- | --- |
| 0 | fold | Give up the hand. |
| 1 | check/call | Check if nothing is owed, otherwise match the current bet. |
| 2 | raise half-pot | Add half the current pot. |
| 3 | raise pot | Add the current pot. |
| 4 | all-in | Put in every remaining chip. |

Fold and check/call are always legal. A raise drops out of the legal list when
the player does not have the chips, or when a half-pot raise would not actually
increase the bet. The trainer rejects an action that is not in `legal_actions`.

The campaign form's **Raises per street** field caps no-limit betting. `0`
(the default) leaves a street uncapped, which is RLCard's own rule. A positive
number counts half-pot, pot, and all-in. Once that many raises have been made,
those three actions leave the legal list until the next card. The blinds are
not raises, and check, call, and fold stay available. Limit and Leduc ignore
this setting.

A hand deals two hole cards each, posts the blinds, then runs preflop, flop
(3 cards), turn (1), and river (1). A street ends when everyone still in the
hand has matched the bet or cannot act. At showdown RLCard compares the best
five-card hands. If everyone but one player folds, that player wins the pot.

Two episode modes already exist on the adapter:

- `hand` — one hand, then stop. The payoff is that hand's chip change. This is
  the mode to use for DQN.
- `session` — keep dealing until someone cannot post the big blind. The payoff
  is the whole session. That single number is a weak training signal, because
  many hands share it.

## Opponents

Three fixed bots live in `spark/agents/no_limit_holdem/bots.py`. They do not
learn. Seat the DQN against any mix of them. The random agent
(`no-limit-random`) is still there too.

| Agent id | Name | What it does |
| --- | --- | --- |
| `no-limit-call-only` | Call Only Agent | Always checks or calls. Never folds and never raises. |
| `no-limit-always-raise` | Always Raise Agent | Takes the largest legal raise: all-in, then pot, then half-pot. Calls when a raise is not legal. |
| `no-limit-strong-hands` | Strong Hands Agent | Raises premium hands and made hands of two pair or better. Calls a pot-sized bet or smaller with a pair, a draw, or a playable preflop hand. Folds weak hands to a bet. |

Premium preflop hands are pairs of tens or better, ace-king, and ace-queen.
Playable hands are any pair, any ace, two cards ten or higher, and suited cards
no more than two ranks apart. After the flop, a straight or better raises the
pot, two pair or trips raises half the pot, and one pair or a four-card draw
calls when the bet is no bigger than the pot.

These bots read `observation["public"]` only. They never see opponent hole cards.

## What the agent receives

The trainer calls:

```python
action = agent.select_action(observation, legal_actions, player_id)
```

`observation` is one dictionary. The DQN reads `observation["public"]` and
ignores the rest. `legal_actions` is a separate list of the ids that are legal
right now. `player_id` is the seat being asked to act.

`observation["encoded"]` is RLCard's older 54-number summary (52 card slots
with hole cards and board mixed together, plus two raw chip counts). It stays
on the dictionary for logs. Do not train on it. It cannot tell a card in the
hand from the same card on the board, and it has no per-seat fold or all-in flag.

`observation["raw"]` is the same hand in RLCard's own keys. `public` is the
stable record built from it. In normal play hidden opponent hole cards are not
in either one. They live only in `environment.hidden_state()`, which is stored
for replay and must not be passed into the network. The one exception is a
seat with **Sees all cards** ticked, described below, which deliberately gets
them in `public["opponent_hole_cards"]`.

`observation["stacks"]` and `observation["my_stack"]` are the session buy-in
totals. They update when a hand ends. During a hand, the chips that matter are
on `public`: the stack still in front of each player, and the chips they have
already put in the pot.

## The `public` record

Seats are in real table order. Seat 0 is always seat 0, even when seat 2 is
the player being asked to act. `encode()` does the rotation.

```text
public = {
    "hole_cards": ["S5", "DA"],   # this player only
    "board": [],                  # 0 to 5 community cards
    "stage": 0,                   # 0 preflop, 1 flop, 2 turn, 3 river
    "pot": 3.0,                   # chips in the pot
    "to_call": 2.0,               # chips this player must add to match the bet
    "big_blind": 2.0,
    "hero_seat": 1,               # seat index of the player to act
    "dealer_button": 1,           # seat index the button sits on this hand
    "num_players": 3,
    "full_vision": False,         # True when this seat has "Sees all cards" ticked
    "opponent_hole_cards": {      # ONLY present when full_vision is True
        "0": ["H9", "C9"],        # seat id -> that seat's two cards
        "2": ["DK", "S4"],
    },
    "seats": [
        {
            "seat_id": 0,
            "stack": 48.0,        # chips not yet in the pot
            "committed": 2.0,     # chips this player has put in this hand
            "status": "active",   # "active", "folded", or "all-in"
            "is_button": False,
        },
        ...
    ],
}
```

Card ids follow RLCard's deck order: suit `S`, `H`, `D`, `C`, then rank
`A, 2, 3, 4, 5, 6, 7, 8, 9, T, J, Q, K`. `SA` is the ace of spades. The full
map is `card2index.json` inside the installed `rlcard` package. `encode()`
loads that file, so the 52 card slots stay aligned with it.

### Button

The button is the dealer marker. It moves one seat clockwise every hand.
`dealer_button` is the seat index it currently occupies. The seat with the
button acts last on the flop, turn, and river. The same cards are a different
decision when the button is on you than when it is on someone else.

### Folded and all-in

RLCard marks a player folded when they choose action 0. It marks a player
all-in when an action spends their last chip. A player who posted a blind that
already spent their whole stack is also treated as all-in here, even if RLCard
has not flipped its own flag yet. Folded wins if both would apply.

Those statuses are the `status` string on each seat in `public`. The agent
never reads RLCard's player object. `encode()` copies the string into two
yes/no numbers in that seat's block of the vector:

- folded → the folded slot is `1` and the all-in slot is `0`
- all-in → the all-in slot is `1` and the folded slot is `0`
- active → both slots are `0`

A folded player is out of the pot and takes no more turns. An all-in player
still contests the pot and also takes no more turns. When it is your turn, your
own folded and all-in slots are `0`. The slots that change your decision are
the other seats.

## The observation vector

`encode_public` returns a list of 222 floats. That length is `FEATURE_DIM` in
`agent.py`. A 2-player hand and a 10-player hand both produce 222 numbers.
Empty seat blocks are zeros. `agent.encode` appends belief features and hero
hand-strength features after this public vector, so the network input is
longer than 222.

Each card is three 0/1 flags: board, hole, and opponent. At most one is 1.
All three zero means the card is unseen. A separate unseen flag would only
repeat that, and a linear layer would otherwise treat "in the hand" as twice
"on the board".

Chip amounts in this vector are divided by the big blind. A pot of 10 chips
with a blind of 2 is stored as `5`. A starting stack of 50 chips is stored as
`25`. The network then sees "5 big blinds in the pot" and "a stack of 25 big
blinds." A short stack and a deep stack stay different. Dividing by the buy-in
instead would turn both into `1.0`.

| Indexes | Contents |
| --- | --- |
| 0–155 | 52 cards × 3 flags. Card `i` starts at `3i`: board, hole, opponent. All zeros means unseen. |
| 156–159 | Street, one-hot. 156 preflop, 157 flop, 158 turn, 159 river. |
| 160 | Pot, in big blinds. |
| 161 | Amount still needed to call, in big blinds. |
| 162–221 | Ten seat blocks of 6 numbers. Block 0 starts at 162. |

Seat block `k` starts at index `162 + 6k`. The six numbers, in order, are:

| Offset | Name | Value |
| --- | --- | --- |
| 0 | occupied | `1` if this block is a real seat, else `0` |
| 1 | stack | remaining chips, in big blinds |
| 2 | committed | chips already in the pot this hand, in big blinds |
| 3 | folded | `1` if that player has folded |
| 4 | all-in | `1` if that player is all-in |
| 5 | button | `1` if the button is on that seat |

Block 0 is always the player who is acting. Block 1 is the next seat in play
order (`hero_seat + 1`, wrapping around the table). Then the rest of the
table. After a 3-player hand, blocks 3 through 9 are zeros.

The player on your left has folded when index 171 is `1`
(`162 + 6 + 3`). That same player is all-in when index 172 is `1`.

The five legal-action flags are not inside these 222 numbers. `forward_pass`
receives `legal_actions` and sets each illegal Q-value to `-1e9` before
`argmax` or `max`. `legal_mask(legal_actions)` returns the same information as
five `0/1` numbers if you want it for debugging.

This vector is a Markov view of the decision. Current cards, stacks,
commitments, who has folded, who is all-in, and where the button sits already
summarize the betting. The action history does not need to go into the network.

The channel offsets are named in `agent.py`: `CARD_BOARD = 0`, `CARD_HOLE = 1`,
`CARD_OPPONENT = 2`. They are positions inside a card's block, not magnitudes.
There is no unseen channel.

### Sees all cards: one seat with perfect information

Every seat on the campaign form has a **Sees all cards** tick box. It is a
property of the seat, not the table, so one learner can be shown everything
while its opponents keep playing real poker. In a seat config it is
`"full_vision": true`; the CLI shortcut for the `--agents` form is
`--full-vision-seats 0`.

- Unticked (default) is normal poker. `public["full_vision"]` is `False`,
  `public` never contains an opponent's cards, and every card that is not the
  hero's hole cards or the board is all zeros. The opponent flag is 0 for
  all 52 cards, so a hidden hole card looks the same as a card still in the deck.
- Ticked: `public["full_vision"]` is `True` and `public["opponent_hole_cards"]`
  maps every other seat id to that seat's two cards. `encode()` turns on the
  opponent flag for those cards. Hero cards stay on the hole flag and board
  cards stay on the board flag.

Use it as a learning check. If a learner cannot beat Call Only even when it can
see every card, the problem is the network or the update, not the hidden
information. Leave the opponents unticked so the test is still "can it exploit
what it knows", and untick the learner again once it clears that bar. The
built-in bots read only their own cards, so ticking them changes nothing.

The vector is 222 numbers either way, so the same saved network loads in both
settings. A learner trained unticked can continue ticked and back without a
new member. A network trained on the old 0/1/2/3 card codes will not load:
the input length changed. With three or more players, the opponent flag says
a card is in some opponent's hand but not whose. Heads-up it is exact. The
environment records the chosen seats as `full_vision_seats` in `hidden_state()`.

## Reward and gamma

Store a transition only for the seat that acted. The next state is the next
time **that same seat** acts. Every opponent action in between shows up as a
new pot, a new board, a fold flag, or the end of the hand.

Immediate rewards:

- Every decision except this seat's last one in the episode: reward `0`,
  `done = False`.
- Last decision: reward `payoff / big_blind`, `done = True`. A win of 10 chips
  with a blind of 2 is `+5`. A loss of 10 chips is `-5`.

`gamma` defaults to `1`. The training target for a non-terminal decision is
therefore the max legal Q-value of the next time this player acts, with nothing
discounted away. The last decision's target is the hand result. Walking
backward, the flop bet and the preflop call are trained on that same result.
A separate bonus for "good" flop play is not used. A correct call can still
lose to a later card, and the chip result at the end of the hand is the score
the agent is supposed to learn.

Use `--episode-mode hand` so that result arrives after a few decisions. A long
cash session paid only when someone busts makes `gamma = 1` unstable.

The Bellman target, once a network exists, is vanilla DQN with illegal actions
masked:

```text
terminal:     y = reward
otherwise:    y = reward + gamma * max(Q_target(next_state) over legal actions)
```

The loss is mean squared error between `y` and the main network's Q-value for
the action that was taken. The target network is a copy of the main network.
Its weights are copied over every `target_update` gradient steps (default
1000). Epsilon decays only after a real gradient step, so random play before
the network exists does not burn the exploration schedule down to the floor.

## Several opponents at one table

Set `--table-size` from 2 to 10. Each seat is its own agent instance, with its
own `seat_id`. A 6-player table can be one `no-limit-dqn` seat in train mode
and five other agents in evaluate mode. The environment does not care which
policy sits in a seat.

Each learning seat stores only its own decisions. The other seats acting in the
same hand means they act before this seat's next state. The observation vector already
has room for them. You do not build a second encoder when the table grows. You
fill more seat blocks.

Two `no-limit-dqn` seats in train mode are two separate networks and two
replay buffers. They do not share weights.

## Agent functions

`NoLimitHoldemDQNAgent` (`agent_id = "no-limit-dqn"`) is discovered
automatically. The random agent stays the default, so existing jobs do not
switch to this agent by accident.

| Function | What it does now |
| --- | --- |
| `encode(observation)` | Builds the public vector from `observation["public"]`, then appends belief and hand-strength features. |
| `terminal_reward(payoff, big_blind)` | Divides the chip result by the big blind. |
| `legal_mask(legal_actions)` | Length-5 `0/1` mask. Not part of the observation vector. |
| `ReplayBuffer.add / sample / purge` | Stores transitions and drops the oldest past `replay_size`. |
| `purge_replay_buffer()` | Trims the buffer to capacity. |
| `forward_pass(state, main, mask_illegal, legal_actions, grad)` | Runs the main or target network. Returns `None` until a network exists. `grad=False` is for action choice and for the target. |
| `pi(observation, legal_actions)` | Epsilon-greedy. A uniform legal action until a network exists. |
| `select_action(...)` | Trainer entry point. Updates the belief bookkeeping, then calls `pi`. |
| `action_probs(...)` | Uniform without a network. Epsilon-greedy probabilities once a network exists. Evaluation with a network is greedy. |
| `make_batch()` | Draws `batch_size` transitions, or returns `[]` if the buffer is shorter than that. |
| `update_Q_value(batch)` | One Adam step on the masked Bellman error. No-op, and no epsilon decay, without a network. |
| `update_Q_target()` | Copies main weights onto the target network. |
| `update(episode)` | Writes this seat's transitions, then calls `update_Q_value` once. |
| `state_dict` / `load_state_dict` | Checkpoint. Replay is left out unless `save_replay` is true, because a full buffer makes the checkpoint file enormous. Network weights are saved once a network exists. |

Those functions, and the defaults below, are on `NoLimitHoldemDQNAgent` in
`dqn.py`. `agent.py` only builds the shared vector.

Defaults: `gamma = 1`, `epsilon = 1`, `epsilon_min = 0.05`, `epsilon_decay = 0.995`
once per gradient step, `lr = 0.001`, `batch_size = 32`, `replay_size = 100000`,
`target_update = 1000`. The trainer does not pass these in. Change the defaults
in `dqn.py` if you need different values. The trainer does set `seat_id`
and `training`.

## The file you write

Edit the four functions at the top of `spark/agents/no_limit_holdem/dqn.py`.
The constants next to them (`HIDDEN_LAYERS`, `BELIEF_FEATURE_DIM`, the action
ids, and the card codes) are for those functions. The trainer does not read
them. Gamma and the other learning settings are on the class in the same file.

| Function | What to return |
| --- | --- |
| `build_q_network(state_dim, action_dim)` | A `torch.nn.Module`. Input shape `(batch, state_dim)`, output shape `(batch, 5)`. |
| `prior(belief, seat, hands)` | `P(hand)` before the latest action. Uniform on the first action. Later, the previous posterior with impossible hands removed, renormalized. |
| `bayes_update(prior, likelihoods, action)` | `P(hand \| action)` proportional to `P(action \| hand) * P(hand)`, summing to 1. If every likelihood is 0, return the prior. |
| `belief_features(belief)` | The list of numbers to append after the public vector. Leave it raising until you want belief in the network. |

`state_dim` is `FEATURE_DIM` (222) plus the hero hand-strength block while
`belief_features` raises. When that function returns a list, `encode`
concatenates it and the network input grows by `len(list)`. Import torch
inside `build_q_network`. The next call to
`select_action` or `update` builds a target copy and an Adam optimizer from
the module you return.

A first network, before belief features exist:

```python
def build_q_network(state_dim: int, action_dim: int):
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(state_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, action_dim),
    )
```

Do not feed `hidden_state` or `observation["encoded"]` into this network.
Do not put the legal-action mask in the inputs. `forward_pass` already masks
illegal Q-values.

## Belief over an opponent's hole cards

Each opponent seat has its own belief. There is no joint distribution over
the whole table. Heads-up, before the flop, the support is the 1,225
two-card hands left after removing this player's two hole cards.

`observation["public"]["action_log"]` is the record the update reads. Each
entry is `{seat, action, legal_actions, stage}` and `legal_actions` is the
set that was legal when that player acted. The first decision of a hand has
an empty log. Opponent hole cards are not in `public`.

On this seat's turn, `select_action` does the following before it picks an
action:

1. If `hand_index` changed, clear `belief` and `likelihoods`.
2. For each new opponent action, enumerate `possible_hole_pairs` from the
   cards this player has seen (hole cards and the board only).
3. For each guessed hand, build a hypothetical public view with that hand
   as the opponent's hole cards and ask `opponent.action_probs`.
4. Call `prior`, then `bayes_update`, and store the result on
   `self.belief[seat]`.

Step 4 is skipped while those functions raise, so a missing posterior does
not stop the hand. `self.likelihoods[seat][hand]` still holds
`{action_id: probability}` after step 3, once an opponent is wired and the
log has one of their actions.

`possible_hole_pairs` lives on the environment. Pass it the seen cards. Do
not remove another opponent's unknown cards from the support; that overlap
is part of the probability you write in `prior` and `bayes_update`.

Belief is per hand and is not written into checkpoints. Rewards are
unchanged: 0 until this seat's last decision, then the chip result divided
by the big blind.

## Running a job

One learner against the Call Only Agent, one hand per episode. In PowerShell:

```powershell
py -m spark.training.cli `
  --environment no-limit-holdem `
  --episode-mode hand `
  --table-size 2 `
  --agents no-limit-dqn no-limit-call-only `
  --episodes 1000 `
  --seed 42 `
  --run-id nl-dqn-demo
```

`--agents` puts every named seat in train mode. The fixed bots ignore training
updates, so that is safe. Swap `no-limit-call-only` for `no-limit-always-raise`,
`no-limit-strong-hands`, or `no-limit-random`. For a larger table, raise
`--table-size` and list one agent per seat.

To mark a seat evaluate-only, pass `--seat-config`. PowerShell removes the
quotes inside a JSON argument unless they are escaped:

```powershell
$seats = '[{"agent_id":"no-limit-dqn","mode":"train"},{"agent_id":"no-limit-always-raise","mode":"evaluate"}]'
py -m spark.training.cli `
  --environment no-limit-holdem `
  --episode-mode hand `
  --table-size 2 `
  --seat-config ($seats.Replace('"','\"')) `
  --episodes 1000 `
  --seed 42 `
  --run-id nl-dqn-demo
```

Metrics `seat_<n>.loss`, `seat_<n>.epsilon`, and `seat_<n>.replay_size` are
written on each training episode. Loss stays `0` and epsilon stays at its
start value until `build_q_network` returns a module.

The same three bots are seeded into the league for `no-limit-holdem` when the
dashboard loads built-in opponents.

Add `--full-vision-seats 0` to the first command to show seat 0 every
opponent's cards. With `--seat-config`, put `"full_vision":true` on the seat
instead.

## Keeping one learner across campaigns

Inside a run the trainer never resets an agent between episodes. It builds one
`NoLimitHoldemDQNAgent` per seat when the run starts and reuses that object for
every hand, so the network, epsilon, the gradient-step counter, and the replay
buffer all carry forward. Only the per-hand belief is cleared.

To keep one learner across *several* runs, make it a **League member in
Training mode**:

1. League tab → **Blank agent**, implementation `No-Limit DQN`, mode
   Training. This writes the agent's fresh `state_dict` to a member file.
2. Campaign tab → seat source **League member**, pick that member. The
   campaign's first checkpoint is built from the member's saved state.
3. While the run plays, the trainer writes the agent's `state_dict` back onto
   the member at every checkpoint interval and when the run stops. Network
   weights, epsilon, and `gradient_steps` are in that state.
4. The next campaign that seats the same member continues from the updated
   weights. Train against the bots first with **Sees all cards** unticked,
   then run the same member again with it ticked on the learner's seat only.

The replay buffer is **not** written to the member (it would make the file
enormous), so a new run starts with an empty buffer and plays `batch_size`
transitions before its first gradient step. Weights and epsilon do carry over.

Two `no-limit-dqn` seats at one table are two separate learners with separate
weights. Seat the same member twice if you want them to start identical.

### Freezing a policy

On the League tab every Training member has a **Freeze** button. It copies the
member's current state into a new member in **Snapshot** mode. The copy plays
greedily (epsilon off) and never updates; the original keeps training. Freeze
before ticking **Sees all cards** so the pure-poker policy is kept, and freeze
again at milestones to build a ladder of opponents for evaluation.

Through the API, Freeze is `POST /api/league/members/{id}/copy` with
`{"name": "...", "mode": "snapshot"}`. The **Save agent** form on the Overview
tab does the same thing from a run's `checkpoint.json` into the Agent library;
import that snapshot into the League to play against it.
