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
| Reporting Protocol | A re-evaluation of the best intermediate agent |
| Number of seeds | 1 |
| Number of common domains | 52 |
| Number of domains where paper scores higher | 17 |
| Number of domains where ChainerRL scores higher | 34 |
| Number of ties between paper and ChainerRL | 1 | 



Comparison against original reported results...
| Game        | ChainerRL Score           | Original Reported Scores |
| ------------- |:-------------:|:-------------:|
| AirRaid | 7637.9| N/A|
| Alien | **14070.0**| 9491.7|
| Amidar | 4863.4| **5131.2**|
| Assault | **15224.3**| 14198.5|
| Asterix | **579136.5**| 428200.3|
| Asteroids | **4822.1**| 2712.8|
| Atlantis | **909249.0**| 826659.5|
| BankHeist | **1625.8**| 1358.0|
| BattleZone | **88680.0**| 62010.0|
| BeamRider | **18645.4**| 16850.2|
| Berzerk | **3616.7**| 2545.6|
| Bowling | **68.3**| 30.0|
| Boxing | **100.0**| 99.6|
| Breakout | 329.5| **417.5**|
| Carnival | 4450.0| N/A|
| Centipede | **8928.9**| 8167.3|
| ChopperCommand | 9681.0| **16654.0**|
| CrazyClimber | 153402.5| **168788.5**|
| Defender | N/A| 55105.0|
| DemonAttack | 104304.1| **111185.2**|
| DoubleDunk | -0.7| **-0.3**|
| Enduro | **2298.7**| 2125.9|
| FishingDerby | **42.0**| 31.3|
| Freeway | 33.5| **34.0**|
| Frostbite | **11324.3**| 9590.5|
| Gopher | **89685.7**| 70354.6|
| Gravitar | **1420.0**| 1419.3|
| Hero | 36184.2| **55887.4**|
| IceHockey | **3.3**| 1.1|
| Jamesbond | 21157.5| N/A|
| JourneyEscape | -63.5| N/A|
| Kangaroo | **15498.5**| 14637.5|
| Krull | 8444.8| **8741.5**|
| KungFuMaster | 36593.5| **52181.0**|
| MontezumaRevenge | 0.0| **384.0**|
| MsPacman | **5593.9**| 5380.4|
| NameThisGame | **13591.4**| 13136.0|
| Phoenix | **261011.3**| 108528.6|
| Pitfall | -0.2| **0.0**|
| Pong | **21.0**| 20.9|
| Pooyan | 12462.4| N/A|
| PrivateEye | 86.0| **4234.0**|
| Qbert | **42774.6**| 33817.5|
| Riverraid | 21264.0| N/A|
| RoadRunner | **63157.0**| 62041.0|
| Robotank | **74.2**| 61.4|
| Seaquest | 1849.8| **15898.9**|
| Skiing | **-9741.2**| -12957.8|
| Solaris | **6649.4**| 3560.3|
| SpaceInvaders | 2848.7| **18789.0**|
| StarGunner | **207200.0**| 127029.0|
| Surround | N/A| 9.7|
| Tennis | **-0.0**| **0.0**|
| TimePilot | **23810.5**| 12926.0|
| Tutankham | **266.7**| 241.0|
| UpNDown | 67551.6| N/A|
| Venture | **12.0**| 5.5|
| VideoPinball | 363333.4| **533936.5**|
| WizardOfWor | **22872.0**| 17862.5|
| YarsRevenge | **111187.8**| 102557.0|
| Zaxxon | 21884.0| **22209.5**|




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

| Training time (in days) across all domains | |
| ------------- |:-------------:|
| Mean        |  13.224 |
| Fastest Domain |11.682 (Phoenix)|
| Slowest Domain | 16.838 (Berzerk)|



