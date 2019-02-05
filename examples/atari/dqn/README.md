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
These results reflect ChainerRL  `v0.5.0`.

| Game        | Score           | Reported Scores |           
| ------------- |:-------------:|:-------------:|
| AirRaid | N/A| N/A|
| Alien | N/A| **3069**|
| Amidar | N/A| **739.5**|
| Assault | N/A| **3359**|
| Asterix | N/A| **6012**|
| Asteroids | N/A| **1629**|
| Atlantis | N/A| **85641**|
| Bank Heist | N/A| **429.7**|
| Battlezone | N/A| **26300**|
| Beamrider | N/A| **6846**|
| Berzerk | N/A| N/A|
| Bowling | N/A| **42.4**|
| Boxing | N/A| **71.8**|
| Breakout | N/A| **401.2**|
| Carnival | N/A| N/A|
| Centipede | N/A| **8309**|
| Chopper Command | N/A| **6687**|
| Crazy Climber | N/A| **114103**|
| Demon Attack | N/A| **9711**|
| Double Dunk | N/A| **-18.1**|
| Elevator Action | N/A| N/A|
| Enduro | N/A| **301.8**|
| Fishing Derby | N/A| **-0.8**|
| Freeway | N/A| **30.3**|
| Frostbite | N/A| **328.3**|
| Gopher | N/A| **8520**|
| Gravitar | N/A| **306.7**|
| H.E.R.O. | N/A| **19950**|
| Ice Hockey | N/A| **-1.6**|
| James Bond 007 | N/A| **576.7**|
| Journey Escape | N/A| N/A|
| Kangaroo | N/A| **6740**|
| Krull | N/A| **3805**|
| Kung-Fu Master | N/A| **23270**|
| Montezuma's Revenge | N/A| **0**|
| Ms. Pac-Man | N/A| **2311**|
| Name This Game | N/A| **7257**|
| Phoenix | N/A| N/A|
| Pitfall II | N/A| N/A|
| Pitfall! | N/A| N/A|
| Pong | N/A| **18.9**|
| Pooyan | N/A| N/A|
| Private Eye | N/A| **1788**|
| Qbert | N/A| **10596**|
| River Raid | N/A| **8316**|
| Road Runner | N/A| **18257**|
| Robot Tank | N/A| **51.6**|
| Seaquest | N/A| **5286**|
| Skiing | N/A| N/A|
| Solaris | N/A| N/A|
| Space Invaders | N/A| **1976**|
| Stargunner | N/A| **57997**|
| Tennis | N/A| **-2.5**|
| Time Pilot | N/A| **5947**|
| Tutankham | N/A| **186.7**|
| Up’n Down | N/A| **8456**|
| Venture | N/A| **380.0**|
| Video Pinball | N/A| **42684**|
| WizardOfWor | N/A| **3393**|
| YarsRevenge | N/A| N/A|
| Zaxxon | N/A| **4977**|

						