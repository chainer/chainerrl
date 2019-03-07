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
- `--gpu`. Specifies the GPU. If you do not have a GPU on your machine, run the example with the option `--gpu -1`. E.g. `python train_dqn.py --gpu -1`.
- `--env`. Specifies the environment. 
- `--render`. Add this option to render the states in a GUI window.
- `--seed`. This option specifies the random seed used.
- `--outdir` This option specifies the output directory to which the results are written.

To view the full list of options, either view the code or run the example with the `--help` option.

## Results
These results reflect ChainerRL  `v0.6.0`.

| Game        | Score           | Reported Scores |           
| ------------- |:-------------:|:-------------:|
| AirRaid | N/A| N/A|
| Alien | N/A| **9491.7**|
| Amidar | N/A| **5131.2**|
| Assault | N/A| **14198.5**|
| Asterix | N/A| **429200.3**|
| Asteroids | N/A| **2712.8**|
| Atlantis | N/A| **826659.5**|
| Bank Heist | N/A| **1358.0**|
| Battlezone | N/A| **62010.0**|
| Beamrider | N/A| **16850.2**|
| Berzerk | N/A| **2545.6**|
| Bowling | N/A| **30.0**|
| Boxing | N/A| **99.6**|
| Breakout | N/A| **417.5**|
| Carnival | N/A| N/A|
| Centipede | N/A| **8167.3**|
| Chopper Command | N/A| **16654.0**|
| Crazy Climber | N/A| **168788.5**|
| Defender | N/A| **55105.0**|
| Demon Attack | N/A| **11185.2**|
| Double Dunk | N/A| **-0.3**|
| Elevator Action | N/A| N/A|
| Enduro | N/A| **2125.9**|
| Fishing Derby | N/A| **31.3**|
| Freeway | N/A| **34.0**|
| Frostbite | N/A| **9590.5**|
| Gopher | N/A| **70354.6**|
| Gravitar | N/A| **1419.3**|
| H.E.R.O. | N/A| **55887.4**|
| Ice Hockey | N/A| **1.1**|
| James Bond 007 | N/A| N/A|
| Journey Escape | N/A| N/A|
| Kangaroo | N/A| **14637.5**|
| Krull | N/A| **8741.5**|
| Kung-Fu Master | N/A| **52181.0**|
| Montezuma's Revenge | N/A| **384.0**|
| Ms. Pac-Man | N/A| **5380.4**|
| Name This Game | N/A| **13136.0**|
| Phoenix | N/A| **108528.6**|
| Pitfall II | N/A| N/A|
| Pitfall! | N/A| **0.0**|
| Pong | N/A| **20.9**|
| Pooyan | N/A| N/A|
| Private Eye | N/A| **4234.0**|
| Qbert | N/A| **33817.5**|
| River Raid | N/A| N/A|
| Road Runner | N/A| **62041.0**|
| Robot Tank | N/A| **61.4**|
| Seaquest | N/A| **15898.9**|
| Skiing | N/A| **-12957.8**|
| Solaris | N/A| **3560.3**|
| Space Invaders | N/A| **18789.0**|
| Stargunner | N/A| **127029.0**|
| Surround | N/A| **9.7**|
| Tennis | N/A| **-0.0**|
| Time Pilot | N/A| **12926.0**|
| Tutankham | N/A| **241.0**|
| Up’n Down | N/A| N/A|
| Venture | N/A| **5.5**|
| Video Pinball | N/A| **533936.5**|
| WizardOfWor | N/A| **17862.5**|
| YarsRevenge | N/A| **102557.0**|
| Zaxxon | N/A| **22209.5**|

						