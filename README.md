# MU-MIMO LoRa C-ADR Simulation Environment

This repository contains a Python simulation environment for evaluating Cognitive Adaptive Data Rate (C-ADR) in multi-user MIMO LoRa networks. The simulator was developed to support experiments on spreading factor selection, spatial interference, channel conditioning, and throughput-reliability tradeoffs in a 4x4 MIMO LoRa setting.

The main goal of the project is to compare a Double Deep Q-Network (DDQN)-based C-ADR agent against representative MIMO-LoRa baselines, including static multi-SF multiplexing, STBC-MIMO, and coherent MIMO multiplexing.

## Project Overview

The simulator models a multi-user LoRa uplink with four transmit streams and four receive antennas. The wireless channel is represented as a flat Rayleigh fading MIMO channel matrix. The environment exposes channel-dependent state features to a DDQN agent, which selects a spreading factor for each transmit antenna.

The agent learns to balance two objectives:

1. Maximize effective throughput by selecting low spreading factors when the channel is favorable.
2. Preserve reliability by avoiding identity collisions in ill-conditioned spatial channels.

The repository includes the simulation environment, physical-layer helper functions, baseline agents, DDQN training code, evaluation scripts, and plotting utilities used to generate paper figures.

## Repository Structure

```text
.
├── agents/
│   ├── baseline_lora.py        # Standard ADR-style baseline
│   ├── competitor_agents.py    # Static Multi-SF, STBC-MIMO, and Coherent MIMO baselines
│   └── dqn_agent.py            # DDQN agent and neural network implementation
├── environment/
│   └── wireless_env.py         # 4x4 MIMO LoRa environment and reward model
├── phy_layer/
│   ├── channel_model.py        # AWGN and Rayleigh channel utilities
│   ├── lora_css.py             # LoRa CSS modulation and demodulation helpers
│   └── mimo_coding.py          # MIMO channel and zero-forcing equalizer utilities
├── config.py                   # Shared LoRa and simulation parameters
├── main.py                     # Simple baseline simulation runner
├── train_ai.py                 # DDQN training script
├── evaluate.py                 # Head-to-head baseline vs trained DQN evaluation
├── evaluate_all.py             # Full SOTA comparison and throughput figure generation
├── evalaute_all2.py            # Instantaneous reward evaluation script
├── visualize_sf.py             # Spreading factor selection distribution figure generator
├── test_channel.py             # LoRa channel test script
├── test_mimo.py                # MIMO spatial multiplexing test script
├── test_visualizer.py          # LoRa chirp visualization script
└── lora_mimo_4x4.pth           # Saved trained DDQN model weights
```

Note: Python cache folders such as `__pycache__` are not required for running the project and can be ignored or removed before publishing the repository.

## Main Components

### C-ADR DDQN Agent

The proposed agent is implemented in `agents/dqn_agent.py`. It uses a fully connected neural network with two hidden layers and a Double-DQN update rule. The action space contains all possible spreading factor assignments for four transmit antennas:

```text
SF_i in {7, 8, 9, 10, 11, 12}
```

This produces:

```text
6^4 = 1296 possible actions
```

Each action is converted into a dictionary of the form:

```python
{
    "sf_ant1": 7,
    "sf_ant2": 7,
    "sf_ant3": 7,
    "sf_ant4": 9
}
```

### Wireless Environment

The main simulation environment is implemented in `environment/wireless_env.py`. It generates a 4x4 flat Rayleigh fading channel and samples the link SNR from a Gaussian distribution representing the simulated agricultural IoT scenario.

The state vector contains 35 features:

1. Real part of the 4x4 channel matrix.
2. Imaginary part of the 4x4 channel matrix.
3. Current SNR.
4. Normalized channel condition number.
5. SNR margin relative to the SF7 sensitivity threshold.

The environment models packet failure using two main mechanisms:

1. Sensitivity failure when the SNR is below the required threshold for the selected spreading factor.
2. Spatial identity collision when the MIMO channel is ill-conditioned and insufficient SF separation is used.

### Baseline Agents

The repository includes several comparison agents in `agents/competitor_agents.py`:

- `Rugini2026Agent`: static multi-SF MIMO-style allocation using different spreading factors across antennas.
- `Ma2021Agent`: STBC-MIMO-style baseline emphasizing reliability and diversity.
- `Rugini2024Agent`: coherent MIMO-style baseline using the same SF across antennas for high-rate multiplexing.

A simple ADR baseline is also included in `agents/baseline_lora.py`.

## Installation

This project was developed with Python 3.12. It should also work with recent Python 3 versions that support PyTorch, NumPy, SciPy, and Matplotlib.

Install the required packages with:

```bash
pip install numpy scipy matplotlib torch
```

For a cleaner setup, create a virtual environment first:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy scipy matplotlib torch
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## How to Run

### 1. Train the DDQN Agent

To train the C-ADR agent from scratch, run:

```bash
python train_ai.py
```

This trains the DDQN agent and saves the model weights as:

```text
lora_mimo_4x4.pth
```

### 2. Evaluate the Trained Agent Against a Baseline

To compare the trained agent against the standard ADR-style baseline, run:

```bash
python evaluate.py
```

### 3. Run the Full Comparative Evaluation

To compare the proposed C-ADR agent against the SOTA-inspired baselines, run:

```bash
python evaluate_all.py
```

This script generates cumulative throughput plots and saves them as:

```text
throughput_comparison_paper.pdf
throughput_comparison_paper.png
```

### 4. Generate the SF Selection Distribution Figure

To analyze the spreading factor selection behavior of the trained policy, run:

```bash
python visualize_sf.py
```

This script saves:

```text
sf_distribution_grouped_paper.pdf
sf_distribution_grouped_paper.png
```

### 5. Run Physical-Layer Tests

To test LoRa modulation and demodulation:

```bash
python test_channel.py
```

To test basic MIMO spatial multiplexing with zero-forcing equalization:

```bash
python test_mimo.py
```

To visualize a LoRa chirp spectrogram:

```bash
python test_visualizer.py
```

## Simulation Assumptions

The current simulator is intended for research prototyping and paper-level comparative analysis. The main assumptions are:

- 4 transmit antennas and 4 receive antennas.
- Flat Rayleigh fading channel.
- Gaussian-distributed SNR with mean 8 dB and standard deviation 3 dB.
- Spreading factors in the set SF7 through SF12.
- Packet failure modeled through sensitivity thresholds and condition-number-based spatial collision rules.
- Throughput modeled as a reward function based on selected spreading factors and implementation-efficiency penalties for selected baselines.

These assumptions are designed to capture the core behavior of channel-aware SF selection in MU-MIMO LoRa, but they do not replace real hardware validation.

## Reproducibility Notes

The repository includes a saved trained model file, `lora_mimo_4x4.pth`, which can be loaded directly by the evaluation scripts. If the model file is removed, `train_ai.py` must be executed before running trained-agent evaluations.

Some scripts generate random Rayleigh channel matrices and random SNR samples. For exact reproducibility, add fixed NumPy and PyTorch random seeds at the beginning of the training and evaluation scripts.

## Known Limitations

The current implementation is simulation-only. It does not yet model all effects present in real LoRa hardware or outdoor agricultural deployments. Important limitations include:

- No SDR or hardware-in-the-loop validation.
- Simplified flat-fading channel model.
- Idealized access to gateway-side channel features.
- No explicit timing, synchronization, carrier frequency offset, or oscillator drift model.
- Simplified packet error and throughput abstractions.
- No direct implementation of a full LoRaWAN MAC layer.

Future work should validate the learned C-ADR policy in a Software Defined Radio testbed and evaluate robustness under imperfect CSI, quantized feedback, hardware non-linearities, foliage attenuation, and time-varying interference.

## AI-Assisted Development Disclosure

Portions of the code development and documentation process were assisted by AI coding tools, including ChatGPT and Gemini. All implementation choices, simulation design, validation, and final repository contents were reviewed and curated by the author.
