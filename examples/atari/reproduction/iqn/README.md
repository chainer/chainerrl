# IQN
This example trains an IQN agent, from the following paper: [Implicit Quantile Networks for Distributional Reinforcement Learning](https://arxiv.org/abs/1806.06923). 

## Requirements

- atari_py>=0.1.1
- opencv-python

## Running the Example

To run the training example:
```
python train_iqn.py [options]
```

We have already pretrained models from this script for all the domains list in the [results](#Results). Note that while we may have run multiple seeds, our pretrained model represents a single run from this script, and may not be representative of the [results](#Results). To load a pretrained model:

```
python train_iqn.py --demo --load-pretrained --env BreakoutNoFrameskip-v4 --pretrained-type best --gpu -1
```

### Useful Options
- `--gpu`. Specifies the GPU. If you do not have a GPU on your machine, run the example with the option `--gpu -1`. E.g. `python train_dqn.py --gpu -1`.
- `--env`. Specifies the environment. 
- `--render`. Add this option to render the states in a GUI window.
- `--seed`. This option specifies the random seed used.
- `--outdir` This option specifies the output directory to which the results are written.
- `--demo`. Runs an evaluation, instead of training the agent.
- `--load-pretrained` Loads the pretrained model. Both `--load` and `--load-pretrained` cannot be used together.
- `--pretrained-type`. Either `best` (the best intermediate network during training) or `final` (the final network after training).

To view the full list of options, either view the code or run the example with the `--help` option.


## Results
These results reflect ChainerRL  `v0.8.0`. We use the same evaluation protocol used in the IQN paper.


| Results Summary ||
| ------------- |:-------------:|
| Reporting Protocol | The highest mean intermediate evaluation score |
| Number of seeds | 3 |
| Number of common domains | 55 |
| Number of domains where paper scores higher | 23 |
| Number of domains where ChainerRL scores higher | 28 |
| Number of ties between paper and ChainerRL | 4 | 


## Evaluation Protocol
Our evaluation protocol is designed to mirror the evaluation protocol of the original paper as closely as possible, in order to offer a fair comparison of the quality of our example implementation. Specifically, the details of our evaluation (also can be found in the code) are the following:

- **Evaluation Frequency**: The agent is evaluated after every 1 million frames (250K timesteps). This results in a total of 200 "intermediate" evaluations.
- **Evaluation Phase**: The agent is evaluated for 500K frames (125K timesteps) in each intermediate evaluation. 
	- **Output**: The output of an intermediate evaluation phase is a score representing the mean score of all completed evaluation episodes within the 125K timesteps. If there is any unfinished episode by the time the 125K timestep evaluation phase is finished, that episode is discarded.
- **Intermediate Evaluation Episode**: 
	- Capped at 30 mins of play, or 108K frames/ 27K timesteps.
	- Each evaluation episode begins with a random number of no-ops (up to 30), where this number is chosen uniformly at random.
- **Reporting**: For each run of our IQN example, we take the best outputted score of the intermediate evaluations to be the evaluation for that agent. We then average this over all runs (i.e. seeds) to produce the output reported in the table.


## Training times

| Training time (in days) across all runs (# domains x # seeds) | |
| ------------- |:-------------:|
| Mean        |  9.104 |
| Standard deviation | 0.486|
| Fastest run | 8.374|
| Slowest run | 10.986|



