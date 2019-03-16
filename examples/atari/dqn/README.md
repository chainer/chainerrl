# DQN
This example trains a DQN agent, from the following paper: [Human-level control through Deep Reinforcement Learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf). 

## Requirements

- atari_py>=0.1.1
- opencv-python

## Running the Example

```
python train_dqn.py [options]
```

### Useful Options
- `--gpu`. Specifies the GPU. If you do not have a GPU on your machine, run the example with the option `--gpu -1`. E.g. `python train_dqn.py --gpu -1`.
- `--env`. Specifies the environment. 
- `--render`. Add this option to render the states in a GUI window.
- `--seed`. This option specifies the random seed used.
- `--outdir` This option specifies the output directory to which the results are written.

To view the full list of options, either view the code or run the example with the `--help` option.

## Results
These results reflect ChainerRL  `v0.5.0`/`v0.6.0`.

The summary of the results is as follows:
 - These results are averaged over 5 runs per domain
 - We ran this example on 59 Atari domains. 
 - The original DQN paper paper ran results on 49 Atari domains. Within these 49 domains the results are as follows:
 	- The reported results from the DQN paper are higher than ChainerRL on 26 domains.
 	- This example from ChainerRL outperforms the DQN paper on 22 domains.
 	- Our implementation ties the reported DQN results on a single domain.
 - Note that the reported DQN results are from a single run on each domain, and might not be an accurate reflection of the DQN's true performance.


| Game        | ChainerRL Score           | Reported Scores |
| ------------- |:-------------:|:-------------:|
| AirRaid | **6450.5**| N/A|
| Alien | 1713.1| **3069**|
| Amidar | **986.7**| 739.5|
| Assault | 3317.2| **3359**|
| Asterix | 5936.7| **6012**|
| Asteroids | 1584.5| **1629**|
| Atlantis | **96456.0**| 85641|
| BankHeist | **645.0**| 429.7|
| BattleZone | 5313.3| **26300**|
| BeamRider | **7042.9**| 6846|
| Berzerk | **707.2**| N/A|
| Bowling | **52.3**| 42.4|
| Boxing | **89.6**| 71.8|
| Breakout | 364.9| **401.2**|
| Carnival | **5222.0**| N/A|
| Centipede | 5112.6| **8309**|
| ChopperCommand | 6170.0| **6687**|
| CrazyClimber | 108472.7| **114103**|
| DemonAttack | 9044.3| **9711**|
| DoubleDunk | **-9.7**| -18.1|
| Enduro | 298.2| **301.8**|
| FishingDerby | **11.6**| -0.8|
| Freeway | 8.1| **30.3**|
| Frostbite | **1093.9**| 328.3|
| Gopher | 8370.0| **8520**|
| Gravitar | **445.7**| 306.7|
| Hero | **20538.7**| 19950|
| IceHockey | -2.4| **-1.6**|
| Jamesbond | **851.7**| 576.7|
| JourneyEscape | **-1894.0**| N/A|
| Kangaroo | **8831.3**| 6740|
| Krull | **6215.0**| 3805|
| KungFuMaster | **27616.7**| 23270|
| MontezumaRevenge | **0.0**| **0.0**|
| MsPacman | **2526.6**| 2311|
| NameThisGame | 7046.5| **7257**|
| Phoenix | **7054.4**| N/A|
| Pitfall | **-28.3**| N/A|
| Pong | **20.1**| 18.9|
| Pooyan | **3118.7**| N/A|
| PrivateEye | 1538.3| **1788**|
| Qbert | 10516.0| **10596**|
| Riverraid | 7784.1| **8316**|
| RoadRunner | **37092.0**| 18257|
| Robotank | 47.4| **51.6**|
| Seaquest | **6075.7**| 5286|
| Skiing | **-13030.2**| N/A|
| Solaris | **1565.1**| N/A|
| SpaceInvaders | 1583.2| **1976**|
| StarGunner | 56685.3| **57997**|
| Tennis | -5.4| **-2.5**|
| TimePilot | 5738.7| **5947**|
| Tutankham | 141.9| **186.7**|
| UpNDown | **11821.5**| 8456|
| Venture | **656.7**| 380.0|
| VideoPinball | 9194.5| **42684**|
| WizardOfWor | 1957.3| **3393**|
| YarsRevenge | **4397.3**| N/A|
| Zaxxon | **5698.7**| 4977|


## Evaluation Protocol

