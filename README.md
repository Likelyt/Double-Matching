# Double Matching under Complementary Preferences

## Abstract
In this paper, we propose a new algorithm for addressing the problem of matching markets with complementary preferences, where agents' preferences are unknown a priori and must be learned from data. The presence of complementary preferences can lead to instability in the matching process, making this problem challenging to solve. To overcome this challenge, we formulate the problem as a bandit learning framework and propose the Multi-agent Multi-type Thompson Sampling (MMTS) algorithm. The algorithm combines the strengths of Thompson Sampling for exploration with a double matching technique to achieve a stable matching outcome. Our theoretical analysis demonstrates the effectiveness of MMTS as it is able to achieve stability at every matching step, satisfies the incentive-compatibility property, and has a sublinear Bayesian regret over time. Our approach provides a useful method for addressing complementary preferences in real-world scenarios.

## Code Structure
```
Double Matching
│   README.md
│
└───Code
|   |   run.sh                      # Multi-Agent Thomspon Sampling (MMTS) run bash file with different parameters
│   │   Main.py                     # The main file to call the MMTS algorithm (MultiAgent.py)
│   │   MutliAgent.py               # The MMTS algorithm
│   │   utils.sh                    # Helper function for the MutliAgent.py
│   │   toy-matching.ipynb          # the toy example for the MMTS algorithm
│
└───log                             # The log file for the MMTS algorithm (generated by the run.sh), including the matching result
│
└───fig                             # The figures for the MMTS algorithm (generated by the run.sh), with the regret and learning parameter figures.
```


## Data and Results

fig and output (models) are stored in fig and log folders, respectively.

## Requirements

- Python 3.6 or above
- Supported packages

## Usage
```bash
bash code/run.sh
```

## Reference

Pls consider cite our [paper](https://arxiv.org/pdf/2301.10230.pdf):
```bash
@article{li2023double,
  title={Double Matching Under Complementary Preferences},
  author={Li, Yuantong and Cheng, Guang and Dai, Xiaowu},
  journal={arXiv preprint arXiv:2301.10230},
  year={2023}
}
```


