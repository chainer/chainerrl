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
- `--gpu`. Specifies the GPU. If you do not have a GPU on your machine, run the example with the option `--gpu -1`. E.g. `python train_dqn.py --gpu -1`.
- `--env`. Specifies the environment. 
- `--render`. Add this option to render the states in a GUI window.
- `--seed`. This option specifies the random seed used.
- `--outdir` This option specifies the output directory to which the results are written.

To view the full list of options, either view the code or run the example with the `--help` option.

## Results
These results reflect ChainerRL  `v0.6.0`. The ChainerRL score currently consists of a single run. The reported results are compared against the scores from the [Noisy Networks Paper](https://arxiv.org/pdf/1706.10295.pdf)

We aim to follow the original paper's evaluation protocol as closely as possible.

| Game        | ChainerRL Score           | Reported Scores |           
| ------------- |:-------------:|:-------------:|
| AirRaid | N/A| N/A|
| Alien | N/A| N/A|
| Amidar | N/A| N/A|
| Assault | N/A| N/A|
| Asterix | N/A | N/A|
| Asteroids | N/A| N/A|
| Atlantis | N/A| N/A|
| Bank Heist | N/A| N/A|
| Battlezone | N/A| N/A|
| Beamrider | N/A | N/A|
| Berzerk | N/A| N/A|
| Bowling | N/A| N/A|
| Boxing | N/A| N/A|
| Breakout | N/A| N/A|
| Carnival | N/A| N/A|
| Centipede | N/A| N/A|
| Chopper Command | N/A| N/A|
| Crazy Climber | N/A| N/A|
| Demon Attack | N/A| N/A|
| Double Dunk | N/A| N/A|
| Elevator Action | N/A| N/A|
| Enduro | N/A| N/A|
| Fishing Derby | N/A| N/A|
| Freeway | N/A| N/A|
| Frostbite | N/A| N/A|
| Gopher | N/A| N/A|
| Gravitar | N/A| N/A|
| H.E.R.O. | N/A| N/A|
| Ice Hockey | N/A| N/A|
| James Bond 007 | N/A| N/A|
| Journey Escape | N/A| N/A|
| Kangaroo | N/A| N/A|
| Krull | N/A| N/A|
| Kung-Fu Master | N/A| N/A|
| Montezuma's Revenge | N/A| N/A|
| Ms. Pac-Man | N/A| N/A|
| Name This Game | N/A| N/A|
| Phoenix | N/A| N/A|
| Pitfall II | N/A| N/A|
| Pitfall! | N/A| N/A|
| Pong | N/A| N/A|
| Pooyan | N/A| N/A|
| Private Eye | N/A| N/A|
| Qbert | N/A| N/A|
| River Raid | N/A| N/A|
| Road Runner | N/A| N/A|
| Robot Tank | N/A| N/A|
| Seaquest | N/A| N/A|
| Skiing | N/A| N/A|
| Solaris | N/A| N/A*|
| Space Invaders | N/A| N/A|
| Stargunner | N/A| N/A|
| Tennis | N/A| N/A|
| Time Pilot | N/A| N/A|
| Tutankham | N/A| N/A|
| Up’n Down | N/A| N/A|
| Venture | N/A| N/A|
| Video Pinball | N/A| N/A|
| WizardOfWor | N/A| N/A|
| YarsRevenge | N/A| N/A|
| Zaxxon | N/A| N/A|

						
