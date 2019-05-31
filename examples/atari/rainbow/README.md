# Rainbow
This example trains a Rainbow agent, from the following paper: [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298). 

## Requirements

- atari_py>=0.1.1
- opencv-python

## Running the Example

```
python train_rainbow.py [options]
```

### Useful Options
- `--gpu`. Specifies the GPU. If you do not have a GPU on your machine, run the example with the option `--gpu -1`. E.g. `python train_rainbow.py --gpu -1`.
- `--env`. Specifies the environment. 
- `--render`. Add this option to render the states in a GUI window.
- `--seed`. This option specifies the random seed used.
- `--outdir` This option specifies the output directory to which the results are written.

To view the full list of options, either view the code or run the example with the `--help` option.

## Results
These results reflect ChainerRL  `v0.6.0`.

| Results Summary ||
| ------------- |:-------------:|
| Number of seeds | 1 |
| Number of common domains | 49 |
| Number of domains where paper scores higher | 20 |
| Number of domains where ChainerRL scores higher | 27 |
| Number of ties between paper and ChainerRL | 2 | 


| Game        | ChainerRL Score           | Original Reported Scores |
| ------------- |:-------------:|:-------------:|
| AirRaid | 6926.1| N/A|
| Alien | 9376.0| **9491.7**|
| Amidar | N/A| 5131.2|
| Assault | **16203.2**| 14198.5|
| Asterix | **674122.5**| 428200.3|
| Asteroids | **20008.5**| 2712.8|
| Atlantis | **938895.5**| 826659.5|
| BankHeist | 1114.3| **1358.0**|
| BattleZone | **103190.0**| 62010.0|
| BeamRider | **20029.4**| 16850.2|
| Berzerk | **6461.2**| 2545.6|
| Bowling | **80.8**| 30.0|
| Boxing | 99.4| **99.6**|
| Breakout | 360.6| **417.5**|
| Carnival | 6050.1| N/A|
| Centipede | **8429.7**| 8167.3|
| ChopperCommand | **19403.5**| 16654.0|
| CrazyClimber | **177331.0**| 168788.5|
| Defender | N/A| 55105.0|
| DemonAttack | 109342.0| **111185.2**|
| DoubleDunk | -6.8| **-0.3**|
| Enduro | 2125.8| **2125.9**|
| FishingDerby | **57.3**| 31.3|
| Freeway | 31.9| **34.0**|
| Frostbite | **10288.5**| 9590.5|
| Gopher | 69889.0| **70354.6**|
| Gravitar | **2437.3**| 1419.3|
| Hero | 37921.8| **55887.4**|
| IceHockey | **6.2**| 1.1|
| Jamesbond | 20242.0| N/A|
| Kangaroo | **14825.0**| 14637.5|
| Krull | 7896.7| **8741.5**|
| KungFuMaster | 32833.5| **52181.0**|
| MontezumaRevenge | 0.0| **384.0**|
| MsPacman | 5223.1| **5380.4**|
| NameThisGame | N/A| 13136.0|
| Phoenix | **280612.8**| 108528.6|
| Pitfall | -2.2| N/A|
| Pitfall! | N/A| 0.0|
| Pong | **20.9**| **20.9**|
| Pooyan | 20962.1| N/A|
| PrivateEye | 100.0| **4234.0**|
| Qbert | **39152.5**| 33817.5|
| Riverraid | 18084.6| N/A|
| RoadRunner | **68956.5**| 62041.0|
| Robotank | **74.3**| 61.4|
| Seaquest | 1836.7| **15898.9**|
| Skiing | **-9714.6**| -12957.8|
| Solaris | **7086.3**| 3560.3|
| SpaceInvaders | 9352.0| **18789.0**|
| StarGunner | **211851.5**| 127029.0|
| Surround | N/A| 9.7|
| Tennis | **-0.0**| **0.0**|
| TimePilot | **27177.0**| 12926.0|
| Tutankham | 161.1| **241.0**|
| UpNDown | 260453.0| N/A|
| Venture | **1359.5**| 5.5|
| VideoPinball | 465601.0| **533936.5**|
| WizardOfWor | **22575.0**| 17862.5|
| YarsRevenge | 80853.9| **102557.0**|
| Zaxxon | **25779.5**| 22209.5|


## Evaluation Protocol
Our evaluation protocol is designed to mirror the evaluation protocol of the original paper as closely as possible, in order to offer a fair comparison of the quality of our example implementation. Specifically, the details of our evaluation (also can be found in the code) are the following:

- **Evaluation Frequency**: The agent is evaluated after every 1 million frames (250K timesteps). This results in a total of 200 "intermediate" evaluations.
- **Evaluation Phase**: The agent is evaluated for 500K frames (125K timesteps) in each intermediate evaluation. 
	- **Output**: The output of an intermediate evaluation phase is a score representing the mean score of all completed evaluation episodes within the 125K timesteps. If there is any unfinished episode by the time the 125K timestep evaluation phase is finished, that episode is discarded.
- **Intermediate Evaluation Episode**: 
	- Capped at 30 mins of play, or 108K frames/ 27K timesteps.
	- Each evaluation episode begins with a random number of no-ops (up to 30), where this number is chosen uniformly at random.
	- During evaluation episodes the agent uses an epsilon-greedy policy, with epsilon=0.001 (original paper does greedy evaluation because noisy networks are used)
- **Reporting**: For each run of our Rainbow example, we take the network weights of the best intermediate agent (i.e. the network weights that achieved the highest intermediate evaluation), and re-evaluate that agent for 200 episodes. In each of these 200 "final evaluation" episodes, the episode is terminated after 30 minutes of play (30 minutes = 1800 seconds * 60 frames-per-second / 4 frames per action = 27000 timesteps). We then output the average of these 200 episodes as the achieved score for the Rainbow agent. The reported value in the table consists of the average of 1 "final evaluation", or 1 run of this Rainbow example.


## Training times

Time statistics...

| Statistic        |            |            |
| ------------- |:-------------:|:-------------:|
| Mean time (in days) across all domains        |  11.8778333366 |
| Fastest Domain |  Assault | 11.1949926343 |
| Slowest Domain |  Krull | 12.4206600419 |

