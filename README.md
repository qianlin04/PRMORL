# Policy-regularized Offline Multi-objective Reinforcement Learning

This is the code implementation for the paper [Policy-regularized Offline Multi-objective Reinforcement Learning].  This implementation is mainly based the codebase of [PD-MORL](https://openreview.net/forum?id=zS9sRyaPFlJ).

## Prerequisites

- **Operating System**: tested on Ubuntu 18.04.
- **Python Version**: >= 3.8.11.
- **PyTorch Version**: >= 1.8.1.
- **MuJoCo** : install mujoco and mujoco-py of version 2.1 by following the instructions in [mujoco-py](<https://github.com/openai/mujoco-py>).
- **Wandb**

## Installation

```
conda env create -f environment.yml
conda activate PRMORL
pip install -e .
```

Install D4MORL environments by

```
cd ../lib/utilities/morl/MOEnvs
pip install -e .
```



## Data Download

- For the MOSB dataset, one only need to successfully install D4RL without any additional downloads.

- For the D4MORL dataset, one can download it by using commands:

  ```
  pip install gdown
  cd ./PRMORL
  gdown --folder https://drive.google.com/drive/folders/1wfd6BwAu-hNLC9uvsI1WPEOmPpLQVT9k?usp=sharing --output data
  ```

  or accessing the [website](https://drive.google.com/drive/folders/1wfd6BwAu-hNLC9uvsI1WPEOmPpLQVT9k?usp=sharing) via the browser.



## Training

- Run the PRMORL for all D4MORL uniform datasets:

  ```
  cd ./PRMORL
  bash run_bash/train_d4morl_allenv.sh
  ```

- Run the PRMORL for all MOSB datasets:

  ```
  cd ./PRMORL
  bash run_bash/train_d4rl_allenv.sh
  ```

- Single experiment for a specific dataset

  ```
  python train_Mujoco_MOOF_TD3_HER.py --env Hopper --dataset_type expert_uniform --dataset 'd4morl' --weight_bc_reward 100 --pref_perturb_theta 1.0 --policy_regularization 'Diffusion-QL'
  ```

  