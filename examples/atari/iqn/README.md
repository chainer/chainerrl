# IQN
This example trains a IQN agent, from the following paper: [Implicit Quantile Networks for Distributional Reinforcement Learning](https://arxiv.org/abs/1806.06923). 

## Requirements

- atari_py>=0.1.1
- opencv-python

## Running the Example

```
python train_iqn.py [options]
```

### Useful Options
- `--gpu`. Specifies the GPU. If you do not have a GPU on your machine, run the example with the option `--gpu -1`. E.g. `python train_dqn.py --gpu -1`.
- `--env`. Specifies the environment. 
- `--render`. Add this option to render the states in a GUI window.
- `--seed`. This option specifies the random seed used.
- `--outdir` This option specifies the output directory to which the results are written.

To view the full list of options, either view the code or run the example with the `--help` option.

## Results
These results reflect ChainerRL  `v0.6.0`. The ChainerRL score currently consists of a single run. The reported results from the IQN paper are also from a single run. We use the same evaluation protocol used in the IQN paper.

| Game        | ChainerRL Score           | Reported Scores |           
| ------------- |:-------------:|:-------------:|
| AirRaid | N/A| N/A|
| Alien | N/A| **7022**|
| Amidar | N/A| **2946**|
| Assault | N/A| **29091**|
| Asterix | **464666.66** | 342016|
| Asteroids | N/A| **2898**|
| Atlantis | N/A| **978200**|
| Bank Heist | N/A| **1416**|
| Battlezone | N/A| **42244**|
| Beamrider | 35525.0 | **42776**|
| Berzerk | N/A| **1053**|
| Bowling | N/A| **86.5**|
| Boxing | N/A| **99.8**|
| Breakout | **738.0**| 734|
| Carnival | N/A| N/A|
| Centipede | N/A| **11561**|
| Chopper Command | N/A| **16836**|
| Crazy Climber | N/A| **179082**|
| Demon Attack | N/A| **53537**|
| Double Dunk | N/A| **5.6**|
| Elevator Action | N/A| N/A|
| Enduro | N/A| **2359**|
| Fishing Derby | N/A| **33.8**|
| Freeway | N/A| **34.0**|
| Frostbite | N/A| **4342**|
| Gopher | N/A| **118365**|
| Gravitar | N/A| **911**|
| H.E.R.O. | N/A| **28386**|
| Ice Hockey | N/A| **0.2**|
| James Bond 007 | N/A| **35108**|
| Journey Escape | N/A| N/A|
| Kangaroo | N/A| **15487**|
| Krull | N/A| **10707**|
| Kung-Fu Master | N/A| **73512**|
| Montezuma's Revenge | N/A| **0**|
| Ms. Pac-Man | N/A| **6349**|
| Name This Game | N/A| **22682**|
| Phoenix | N/A| **56599**|
| Pitfall II | N/A| N/A|
| Pitfall! | N/A| **0.0**|
| Pong | **21.0**| **21.0**|
| Pooyan | N/A| N/A|
| Private Eye | N/A| **200**|
| Qbert | **25971.3**| 25750|
| River Raid | N/A| **17765**|
| Road Runner | N/A| **57900**|
| Robot Tank | N/A| **62.5**|
| Seaquest | **31670.0**| 30140|
| Skiing | N/A| **-9289**|
| Solaris | N/A| **8007**|
| Space Invaders | **43649.0**| 28888|
| Stargunner | N/A| **74677**|
| Tennis | N/A| **23.6**|
| Time Pilot | N/A| **12236**|
| Tutankham | N/A| **293**|
| Up’n Down | N/A| **88148**|
| Venture | N/A| **1318**|
| Video Pinball | N/A| **698045**|
| WizardOfWor | N/A| **31190**|
| YarsRevenge | N/A| **28379**|
| Zaxxon | N/A| **21772**|

						
