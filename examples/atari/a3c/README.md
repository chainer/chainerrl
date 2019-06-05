# A3C
This example trains an Asynchronous Advantage Actor Critic (A3C) agent, from the following paper: [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783). 

## Requirements

- atari_py>=0.1.1
- opencv-python

## Running the Example

```
python train_a3c.py [options]
```

### Useful Options
- `--gpu`. Specifies the GPU. If you do not have a GPU on your machine, run the example with the option `--gpu -1`. E.g. `python train_a3c.py --gpu -1`.
- `--env`. Specifies the environment. 
- `--render`. Add this option to render the states in a GUI window.
- `--seed`. This option specifies the random seed used.
- `--outdir` This option specifies the output directory to which the results are written.

To view the full list of options, either view the code or run the example with the `--help` option.

## Results
These results reflect ChainerRL  `v0.6.0`. The ChainerRL score currently consists of a single run. The reported results are compared against the scores from the [Noisy Networks Paper](https://arxiv.org/abs/1706.10295), since the original paper does not report scores for the no-op evaluation protocol.

**NOTE: These benchmark scores below come from running train_a3c.py and evaluating every 1 million timesteps, as opposed to every 250K timesteps. New benchmark results will come soon.**

| Results Summary ||
| ------------- |:-------------:|
| Number of seeds | 1 |
| Number of common domains | 52 |
| Number of domains where paper scores higher | 25 |
| Number of domains where ChainerRL scores higher | 24 |
| Number of ties between paper and ChainerRL | 3 | 


| Game        | ChainerRL Score           | Original Reported Scores |
| ------------- |:-------------:|:-------------:|
| AirRaid | 4625.9| N/A|
| Alien | 1397.2| **2027**|
| Amidar | **1110.8**| 904|
| Assault | **5821.6**| 2879|
| Asterix | 6820.7| **6822**|
| Asteroids | 2428.8| **2544**|
| Atlantis | **732425.0**| 422700|
| BankHeist | **1308.9**| 1296|
| BattleZone | 5421.1| **16411**|
| BeamRider | 8493.4| **9214**|
| Berzerk | **1594.2**| 1022|
| Bowling | 31.7| **37**|
| Boxing | **98.1**| 91|
| Breakout | **533.6**| 496|
| Carnival | 5132.9| N/A|
| Centipede | 4849.9| **5350**|
| ChopperCommand | 4881.0| **5285**|
| CrazyClimber | 124400.0| **134783**|
| Defender | N/A| 52917.0|
| DemonAttack | **108832.5**| 37085|
| DoubleDunk | 1.5| **3**|
| Enduro | **0.0**| **0**|
| FishingDerby | **36.3**| -7|
| Freeway | **0.0**| **0**|
| Frostbite | **313.6**| 288|
| Gopher | **8746.5**| 7992|
| Gravitar | 228.0| **379**|
| Hero | **36892.5**| 30791|
| IceHockey | -4.6| **-2**|
| JamesBond | N/A| 509.0|
| Jamesbond | 370.1| N/A|
| JourneyEscape | -871.2| N/A|
| Kangaroo | 115.8| **1166**|
| Krull | **10601.4**| 9422|
| KungFuMaster | **40970.4**| 37422|
| MontezumaRevenge | 1.9| **14**|
| MsPacman | **2498.0**| 2436|
| NameThisGame | 6597.0| **7168**|
| Phoenix | **42654.5**| 9476|
| Pitfall | -10.8| N/A|
| Pitfall! | N/A| 0.0|
| Pong | **20.9**| 7|
| Pooyan | 4067.9| N/A|
| PrivateEye | 376.1| **3781**|
| Qbert | 15610.6| **18586**|
| Riverraid | 13223.3| N/A|
| RoadRunner | 39897.8| **45315**|
| Robotank | 2.9| **6**|
| Seaquest | **1786.5**| 1744|
| Skiing | -16090.5| **-12972**|
| Solaris | 3157.8| **12380**|
| SpaceInvaders | **1630.6**| 1034|
| StarGunner | **57943.2**| 49156|
| Surround | N/A| -8.0|
| Tennis | **-0.3**| -6|
| TimePilot | 3850.6| **10294**|
| Tutankham | **331.4**| 213|
| UpNDown | 17952.0| **89067**|
| Venture | **0.0**| **0**|
| VideoPinball | **407331.2**| 229402|
| WizardOfWor | 2800.0| **8953**|
| YarsRevenge | **25175.5**| 21596|
| Zaxxon | 80.7| **16544**|


## Evaluation Protocol

Our evaluation protocol is designed to mirror the evaluation protocol from the [Noisy Networks Paper](https://arxiv.org/abs/1706.10295) as closely as possible, since the original A3C paper does not report reproducible results (they use human starts trajectories which are not publicly available). The reported results are from the [Noisy Networks Paper](https://arxiv.org/abs/1706.10295), Table 3.

Our evaluation protocol is designed to mirror the evaluation protocol of the original paper as closely as possible, in order to offer a fair comparison of the quality of our example implementation. Specifically, the details of our evaluation (also can be found in the code) are the following:

- **Evaluation Frequency**: The agent is evaluated after every 1 million frames (250K timesteps). This results in a total of 200 "intermediate" evaluations.
- **Evaluation Phase**: The agent is evaluated for 500K frames (125K timesteps) in each intermediate evaluation. 
	- **Output**: The output of an intermediate evaluation phase is a score representing the mean score of all completed evaluation episodes within the 125K timesteps. If there is any unfinished episode by the time the 125K timestep evaluation phase is finished, that episode is discarded.
- **Intermediate Evaluation Episode**: 
	- Each intermediate evaluation episode is capped in length at 27K timesteps or 108K frames.
	- Each evaluation episode begins with a random number of no-ops (up to 30), where this number is chosen uniformly at random.
- **Reporting**: For each run of our A3C example, we report the highest scores amongst each of the intermediate evaluation phases. This differs from the original A3C paper which states that: "We additionally used the final network weights for evaluation". This is because the [Noisy Networks Paper](https://arxiv.org/abs/1706.10295) states that "Per-game maximum scores are computed by taking the maximum raw scores of the agent and then averaging over three seeds".


## Training times

We trained with 17 CPUs and no GPU. However, we used 16 processes (as per the A3C paper).


| Statistic        |            |            |
| ------------- |:-------------:|:-------------:|
| Mean time (in days) across all domains        |  1.08299383309 |
| Fastest Domain |  DemonAttack | 0.736027011088 |
| Slowest Domain |  UpNDown | 1.25626688715 |

				