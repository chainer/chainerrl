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

We use the best intermediate scores on each domain to evaluate A3C.

We aim to follow the evaluation protocol from the [Noisy Networks Paper](https://arxiv.org/abs/1706.10295) as closely as possible. The reported results are from the [Noisy Networks Paper](https://arxiv.org/abs/1706.10295), Table 3.

| Results Summary ||
| ------------- |:-------------:|
| Number of seeds | 1 |
| Number of common domains | 52 |
| Number of domains where paper scores higher | 25 |
| Number of domains where ChainerRL scores higher | 24 |
| Number of ties between paper and ChainerRL | 3 | 



| Game        | ChainerRL Time           |
| ------------- |:-------------:|:-------------:|
| AirRaid | 1.17388990544 days |
| Alien | 1.18438464669 days |
| Amidar | 1.19360730418 days |
| Assault | 1.146660133 days |
| Asterix | 0.975553301926 days |
| Asteroids | 1.18414117109 days |
| Atlantis | 0.753091469528 days |
| BankHeist | 0.769557803519 days |
| BattleZone | 0.792063550681 days |
| BeamRider | 0.75736545874 days |
| Berzerk | 0.756263140633 days |
| Bowling | 1.16921226389 days |
| Boxing | 1.24609514639 days |
| Breakout | 1.16726391949 days |
| Carnival | 1.01271149436 days |
| Centipede | 1.18564387133 days |
| ChopperCommand | 1.22278479479 days |
| CrazyClimber | 0.742167222315 days |
| DemonAttack | 0.736027011088 days |
| DoubleDunk | 0.785350624331 days |
| Enduro | 1.02262304159 days |
| FishingDerby | 0.777940351574 days |
| Freeway | 1.14676468119 days |
| Frostbite | 1.19256961678 days |
| Gopher | 1.14953342136 days |
| Gravitar | 1.14965298631 days |
| Hero | 0.768612422162 days |
| IceHockey | 1.02160934827 days |
| Jamesbond | 1.16947005497 days |
| JourneyEscape | 0.755416593844 days |
| Kangaroo | 1.16526888852 days |
| Krull | 1.05267939805 days |
| KungFuMaster | 1.16863955053 days |
| MontezumaRevenge | 1.17643886358 days |
| MsPacman | 1.19964800659 days |
| NameThisGame | 0.746000485056 days |
| Phoenix | 1.04935906721 days |
| Pitfall | 1.20174458291 days |
| Pong | 1.19954795655 days |
| Pooyan | 1.19573762902 days |
| PrivateEye | 1.17862031261 days |
| Qbert | 1.20038552089 days |
| Riverraid | 1.1646893171 days |
| RoadRunner | 1.20581418721 days |
| Robotank | 1.24274204857 days |
| Seaquest | 1.23936292961 days |
| Skiing | 1.23823434896 days |
| Solaris | 1.24228741978 days |
| SpaceInvaders | 1.13197622961 days |
| StarGunner | 1.1219968725 days |
| Tennis | 1.19821833329 days |
| TimePilot | 1.10073837652 days |
| Tutankham | 1.21187065374 days |
| UpNDown | 1.25626688715 days |
| Venture | 1.17066221943 days |
| VideoPinball | 1.19901385698 days |
| WizardOfWor | 1.15513805685 days |
| YarsRevenge | 1.2318219768 days |
| Zaxxon | 1.14370542539 days |


## Training times

We trained with 17 CPUs and no GPU. However, we used 16 processes (as per the A3C paper).


| Statistic        |            |            |
| ------------- |:-------------:|:-------------:|
| Mean time (in days) across all domains        |  1.08299383309 |
| Fastest Domain |  DemonAttack | 0.736027011088 |
| Slowest Domain |  UpNDown | 1.25626688715 |

				