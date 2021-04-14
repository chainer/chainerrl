# A3C
This example trains an Asynchronous Advantage Actor Critic (A3C) agent, from the following paper: [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783). 

## Requirements

- atari_py>=0.1.1
- opencv-python

## Running the Example

To run the training example:
```
python train_a3c.py [options]
```

We have already trained models from this script for all the domains list in the [results](#Results). To load a pretrained model:
```
python train_a3c.py --demo --load-pretrained --env BreakoutNoFrameskip-v4
```

### Useful Options
- `--env`. Specifies the environment. 
- `--render`. Add this option to render the states in a GUI window.
- `--seed`. This option specifies the random seed used.
- `--outdir` This option specifies the output directory to which the results are written.
- `--demo`. Runs an evaluation, instead of training the agent.
- `--load-pretrained` Loads the pretrained model. Both `--load` and `--load-pretrained` cannot be used together.

To view the full list of options, either view the code or run the example with the `--help` option.


## Results
These results reflect ChainerRL  `v0.8.0`. The reported results are compared against the scores from the [Noisy Networks Paper](https://arxiv.org/abs/1706.10295), since the original paper does not report scores for the no-op evaluation protocol.

| Results Summary ||
| ------------- |:-------------:|
| Reporting Protocol | The highest mean intermediate evaluation score |
| Number of seeds | 5 |
| Number of common domains | 54 |
| Number of domains where paper scores higher | 24 |
| Number of domains where ChainerRL scores higher | 27 |
| Number of ties between paper and ChainerRL | 3 | 


| Game        | ChainerRL Score           | Original Reported Scores |
| ------------- |:-------------:|:-------------:|
| AirRaid | 3923.8| N/A|
| Alien | 2005.4| **2027**|
| Amidar | 869.7| **904**|
| Assault | **6832.6**| 2879|
| Asterix | **9363.0**| 6822|
| Asteroids | **2775.6**| 2544|
| Atlantis | **836040.0**| 422700|
| BankHeist | **1321.6**| 1296|
| BattleZone | 7998.0| **16411**|
| BeamRider | 9044.4| **9214**|
| Berzerk | **1166.8**| 1022|
| Bowling | 31.3| **37**|
| Boxing | **96.0**| 91|
| Breakout | **569.9**| 496|
| Carnival | 4643.3| N/A|
| Centipede | **5352.4**| 5350|
| ChopperCommand | **6997.1**| 5285|
| CrazyClimber | 121146.1| **134783**|
| Defender | N/A| 52917|
| DemonAttack | **111339.2**| 37085|
| DoubleDunk | 1.5| **3**|
| Enduro | 0.0| 0|
| FishingDerby | **38.7**| -7|
| Freeway | 0.0| 0|
| Frostbite | **288.2**| 288|
| Gopher | **9251.0**| 7992|
| Gravitar | 244.5| **379**|
| Hero | **36599.2**| 30791|
| IceHockey | -4.5| **-2**|
| Jamesbond | 376.9| **509**|
| JourneyEscape | -989.2| N/A|
| Kangaroo | 252.0| **1166**|
| Krull | 8949.3| **9422**|
| KungFuMaster | **39676.3**| 37422|
| MontezumaRevenge | 2.8| **14**|
| MsPacman | **2552.9**| 2436|
| NameThisGame | **8646.0**| 7168|
| Phoenix | **38428.3**| 9476|
| Pitfall | -4.4| **0**|
| Pong | **20.7**| 7|
| Pooyan | 4237.9| N/A|
| PrivateEye | 449.0| **3781**|
| Qbert | **18889.2**| 18586|
| Riverraid | 12683.5| N/A|
| RoadRunner | 40660.6| **45315**|
| Robotank | 3.1| **6**|
| Seaquest | **1785.6**| 1744|
| Skiing | -13094.2| **-12972**|
| Solaris | 3784.2| **12380**|
| SpaceInvaders | **1568.9**| 1034|
| StarGunner | **60348.7**| 49156|
| Surround | N/A| -8|
| Tennis | -12.2| **-6**|
| TimePilot | 4506.6| **10294**|
| Tutankham | **296.7**| 213|
| UpNDown | **95014.6**| 89067|
| Venture | 0.0| 0|
| VideoPinball | **377939.3**| 229402|
| WizardOfWor | 2518.7| **8953**|
| YarsRevenge | 19663.9| **21596**|
| Zaxxon | 78.9| **16544**|


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


| Training time (in days) across runs (# domains x 4 seeds) | |
| ------------- |:-------------:|
| Mean        |  0.84 |
| Standard deviation | 0.194|
| Fastest run | 0.666|
| Slowest run | 1.46|

**Note**: These training times represent the training times for only 4 seeds instead of 5 (due to an experimental error).

				