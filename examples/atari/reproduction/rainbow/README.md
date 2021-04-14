# Rainbow
This example trains a Rainbow agent, from the following paper: [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298). 

## Requirements

- atari_py>=0.1.1
- opencv-python

## Running the Example

To run the training example:
```
python train_rainbow.py [options]
```

We have already pretrained models from this script for all the domains list in the [results](#Results) section. To load a pretrained model:

```
python train_rainbow.py --demo --load-pretrained --env BreakoutNoFrameskip-v4 --pretrained-type best --gpu -1
```

### Useful Options
- `--gpu`. Specifies the GPU. If you do not have a GPU on your machine, run the example with the option `--gpu -1`. E.g. `python train_rainbow.py --gpu -1`.
- `--env`. Specifies the environment. 
- `--render`. Add this option to render the states in a GUI window.
- `--seed`. This option specifies the random seed used.
- `--outdir` This option specifies the output directory to which the results are written.
- `--demo`. Runs an evaluation, instead of training the agent.
- `--load-pretrained` Loads the pretrained model. Both `--load` and `--load-pretrained` cannot be used together.
- `--pretrained-type`. Either `best` (the best intermediate network during training) or `final` (the final network after training).

To view the full list of options, either view the code or run the example with the `--help` option.


## Results
These results reflect ChainerRL  `v0.8.0`.

| Results Summary ||
| ------------- |:-------------:|
| Reporting Protocol | A re-evaluation of the best intermediate agent |
| Number of seeds | 3 |
| Number of common domains | 52 |
| Number of domains where paper scores higher | 17 |
| Number of domains where ChainerRL scores higher | 34 |
| Number of ties between paper and ChainerRL | 1 | 


| Game        | ChainerRL Score           | Original Reported Scores |
| ------------- |:-------------:|:-------------:|
| AirRaid | 6754.3| N/A|
| Alien | **11255.4**| 9491.7|
| Amidar | 3302.3| **5131.2**|
| Assault | **17040.6**| 14198.5|
| Asterix | **440208.0**| 428200.3|
| Asteroids | **3274.9**| 2712.8|
| Atlantis | **895215.8**| 826659.5|
| BankHeist | **1655.1**| 1358.0|
| BattleZone | **87015.0**| 62010.0|
| BeamRider | **26672.1**| 16850.2|
| Berzerk | **17043.4**| 2545.6|
| Bowling | **55.7**| 30.0|
| Boxing | **99.8**| 99.6|
| Breakout | 353.0| **417.5**|
| Carnival | 4762.8| N/A|
| Centipede | **8220.1**| 8167.3|
| ChopperCommand | **103942.2**| 16654.0|
| CrazyClimber | **174438.8**| 168788.5|
| Defender | N/A| 55105.0|
| DemonAttack | 101076.9| **111185.2**|
| DoubleDunk | -1.0| **-0.3**|
| Enduro | **2278.6**| 2125.9|
| FishingDerby | **44.6**| 31.3|
| Freeway | 33.6| **34.0**|
| Frostbite | **10071.6**| 9590.5|
| Gopher | **82497.8**| 70354.6|
| Gravitar | **1605.6**| 1419.3|
| Hero | 27830.8| **55887.4**|
| IceHockey | **5.7**| 1.1|
| Jamesbond | 24997.6| N/A|
| JourneyEscape | -429.2| N/A|
| Kangaroo | 11038.8| **14637.5**|
| Krull | 8237.9| **8741.5**|
| KungFuMaster | 33628.2| **52181.0**|
| MontezumaRevenge | 16.2| **384.0**|
| MsPacman | **5780.6**| 5380.4|
| NameThisGame | **14236.4**| 13136.0|
| Phoenix | 84659.6| **108528.6**|
| Pitfall | -3.2| **0.0**|
| Pong | **21.0**| 20.9|
| Pooyan | 7772.7| N/A|
| PrivateEye | 99.3| **4234.0**|
| Qbert | **41819.6**| 33817.5|
| Riverraid | 26574.2| N/A|
| RoadRunner | **65579.3**| 62041.0|
| Robotank | **75.6**| 61.4|
| Seaquest | 3708.5| **15898.9**|
| Skiing | **-10270.9**| -12957.8|
| Solaris | **8113.0**| 3560.3|
| SpaceInvaders | 17902.6| **18789.0**|
| StarGunner | **188384.2**| 127029.0|
| Surround | N/A| 9.7|
| Tennis | -0.0| 0.0|
| TimePilot | **24385.2**| 12926.0|
| Tutankham | **243.2**| 241.0|
| UpNDown | 291785.9| N/A|
| Venture | **1462.3**| 5.5|
| VideoPinball | 477238.7| **533936.5**|
| WizardOfWor | **20695.0**| 17862.5|
| YarsRevenge | 86609.9| **102557.0**|
| Zaxxon | **24107.5**| 22209.5|




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

| Training time (in days) across all runs (# domains x # seeds) | |
| ------------- |:-------------:|
| Mean        |  12.267 |
| Standard deviation | 1.041|
| Fastest run | 10.067|
| Slowest run | 13.974|



