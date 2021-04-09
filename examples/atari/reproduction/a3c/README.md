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


| Training time (in days) across all domains | |
| ------------- |:-------------:|
| Mean        |  1.158 |
| Fastest Domain |1.008 (Asteroids)|
| Slowest Domain | 1.46 (ChopperCommand)|

				