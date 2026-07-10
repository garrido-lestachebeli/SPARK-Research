# <u> SPARK-Research</u>
Implementing Bayes Rule to Reinforcement Learning in Stoccastic Paritally Observable Environments

## <u>Environments</u>

## Blackjack
```
# Run 5 episodes, with a random policy and save them to the default episodes.json
py Blackjack/main.py --count 5 --seed 42 --save
```
Episodes are saved to episodes.json, and replay.py lets you preview the episodes and see what actions were taken

```
#load the generic episodes.json
py Blackjack/replay.py 
py Blackjack/replay.py Blackjack/custom.json
```

Episodes are stored in json format, here is an example episode:


```
{
      "seed": null,
      "initial_state": {
        "player_sum": 17,
        "dealer_showing": 10,
        "usable_ace": true
      },
      "steps": [
        {
          "step": 0,
          "action": 0,
          "action_name": "stick",
          "state_before": {
            "player_sum": 17,
            "dealer_showing": 10,
            "usable_ace": true
          },
          "state_after": {
            "player_sum": 17,
            "dealer_showing": 10,
            "usable_ace": true
          },
          "reward": -1.0,
          "terminated": true,
          "truncated": false,
          "info": {}
        }
      ],
      "total_reward": -1.0,
      "num_steps": 1,
      "terminated": true,
      "truncated": false,
      "outcome": "loss",
      "final_state": {
        "player_sum": 17,
        "dealer_showing": 10,
        "usable_ace": true
      }
    }
```

Next steps: 
- implement policy class which will be passed into main.py to determine which actions are taken
- implement a training class which will handle updating the policy given a new episode


## Leduc Poker

## Limit Texas Holdem

## No Limit Texas Holdem
