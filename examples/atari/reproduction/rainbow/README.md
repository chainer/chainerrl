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
These results reflect ChainerRL  `v0.7.0`.

| Results Summary ||
| ------------- |:-------------:|
| Number of seeds | 1 |
| Number of common domains | 51 |
| Number of domains where paper scores higher | 21 |
| Number of domains where ChainerRL scores higher | 29 |
| Number of ties between paper and ChainerRL | 1 |


| Game        | ChainerRL Score           | Original Reported Scores |
| ------------- |:-------------:|:-------------:|
| AirRaid | 8447.9| N/A|
| Alien | **12163.2**| 9491.7|
| Amidar | 4697.4| **5131.2**|
| Assault | **18425.9**| 14198.5|
| Asterix | 298025.0| **428200.3**|
| Asteroids | **5131.2**| 2712.8|
| Atlantis | **851950.0**| 826659.5|
| BankHeist | **1630.5**| 1358.0|
| BattleZone | **98923.1**| 62010.0|
| BeamRider | **19279.4**| 16850.2|
| Berzerk | **3757.2**| 2545.6|
| Bowling | **45.0**| 30.0|
| Boxing | **99.8**| 99.6|
| Breakout | 351.8| **417.5**|
| Carnival | 4446.0| N/A|
| Centipede | **8337.6**| 8167.3|
| ChopperCommand | 9068.4| **16654.0**|
| CrazyClimber | 163036.0| **168788.5**|
| Defender | N/A| 55105.0|
| DemonAttack | 104041.3| **111185.2**|
| DoubleDunk | **0.0**| -0.3|
| Enduro | **2311.3**| 2125.9|
| FishingDerby | **40.9**| 31.3|
| Freeway | 33.3| **34.0**|
| Frostbite | **10497.6**| 9590.5|
| Gopher | **98084.0**| 70354.6|
| Gravitar | 1302.5| **1419.3**|
| Hero | 30907.1| **55887.4**|
| IceHockey | **2.9**| 1.1|
| Jamesbond | 21323.2| N/A|
| JourneyEscape | -185.3| N/A|
| Kangaroo | **15500.0**| 14637.5|
| Krull | 6761.6| **8741.5**|
| KungFuMaster | 39858.3| **52181.0**|
| MontezumaRevenge | 0.0| **384.0**|
| MsPacman | **6015.4**| 5380.4|
| NameThisGame | 13092.1| **13136.0**|
| Phoenix | **223676.3**| 108528.6|
| Pitfall | -3.5| N/A|
| Pitfall! | N/A| 0.0|
| Pong | **21.0**| 20.9|
| Pooyan | 7946.6| N/A|
| PrivateEye | 100.0| **4234.0**|
| Qbert | **38605.7**| 33817.5|
| Riverraid | 22309.8| N/A|
| RoadRunner | **64002.0**| 62041.0|
| Robotank | **74.5**| 61.4|
| Seaquest | 1843.6| **15898.9**|
| Skiing | **-11093.2**| -12957.8|
| Solaris | 911.8| **3560.3**|
| SpaceInvaders | 2812.9| **18789.0**|
| StarGunner | **202136.4**| 127029.0|
| Surround | N/A| 9.7|
| Tennis | **0.0**| **0.0**|
| TimePilot | **23123.8**| 12926.0|
| Tutankham | **250.7**| 241.0|
| UpNDown | 27630.0| N/A|
| Venture | 0.0| **5.5**|
| VideoPinball | 438907.9| **533936.5**|
| WizardOfWor | **20770.2**| 17862.5|
| YarsRevenge | 101023.5| **102557.0**|
| Zaxxon | 14635.1| **22209.5**|



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

| Training time (in days) across all domains        |            |
| ------------- |:-------------:|
| Mean        |  13.2241181406 |
| Fastest Domain |11.6815262606 (Phoenix)|
| Slowest Domain | 16.8376358549 (Berzerk)|


