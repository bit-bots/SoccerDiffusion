<p align="center">
  <a href="#" style="
    display: inline-block;
    padding: 12px 24px;
    margin: 6px;
    font-size: 18px;
    font-weight: bold;
    color: white;
    background-color: #6c757d;
    border-radius: 8px;
    text-decoration: none;
  ">📄 Paper</a>

  <a href="https://github.com/bit-bots/SoccerDiffusion" style="
    display: inline-block;
    padding: 12px 24px;
    margin: 6px;
    font-size: 18px;
    font-weight: bold;
    color: white;
    background-color: #007bff;
    border-radius: 8px;
    text-decoration: none;
  ">💻 Code</a>

  <a href="https://data.bit-bots.de/SoccerDiffusion/dataset/robocup_2024_german_open_2025.sqlite3.gz" style="
    display: inline-block;
    padding: 12px 24px;
    margin: 6px;
    font-size: 18px;
    font-weight: bold;
    color: white;
    background-color: #28a745;
    border-radius: 8px;
    text-decoration: none;
  ">🗃️ Dataset</a>
</p>

## Abstract

This paper introduces *SoccerDiffusion*, a transformer-based diffusion model designed to learn end-to-end control policies for humanoid robot soccer directly from real-world gameplay recordings.
Using data collected from RoboCup competitions, the model predicts joint command trajectories from multi-modal sensor inputs, including vision, proprioception, and game state.
We employ a distillation technique to enable real-time inference on embedded platforms that reduces the multi-step diffusion process to a single step.
Our results demonstrate the model’s ability to replicate complex motion behaviors such as walking, kicking, and fall recovery both in simulation and on physical robots.
Although high-level tactical behavior remains limited, this work provides a robust foundation for subsequent reinforcement learning or preference optimization methods.

## Walking

<video width="100%" src="assets/walk.mp4" controls muted loop autoplay playsinline>
</video>

## Fall and Standup

<video width="100%" src="assets/get_up_fall_back.mp4" controls muted loop autoplay playsinline ></video>

<video width="100%" src="assets/get_up_fall_right.mp4" controls muted loop autoplay playsinline ></video>

<video width="100%" src="assets/get_up_fall_front.mp4" controls muted loop autoplay playsinline ></video>

## Architecture

![image](architecture.png)

## Acknowledgements

We gratefully acknowledge funding and support from the project [*Digital and Data Literacy in Teaching Lab (DDLitLab)*](https://www.hcl.uni-hamburg.de/ddlitlab.html) at the University of Hamburg and the [*Stiftung Innovation in der Hochschullehre*](https://stiftung-hochschullehre.de/) foundation.
We extend our special thanks to the members of the [*Hamburg Bit-Bots*](https://bit-bots.de/) RoboCup team for their continuous support and for providing data and computational resources.
We also thank the RoboCup teams [*B-Human*](https://b-human.de/) and [*HULKs*](https://hulks.de/) for generously sharing their data for this research.
Additionally, we are grateful to the [*Technical Aspects of Multimodal Systems (TAMS)*](https://tams.informatik.uni-hamburg.de/) research group at the University of Hamburg for providing computational resources.
This research was partially funded by the Ministry of Science, Research and Equalities of Hamburg, as well as the German Research Foundation (DFG) and the National Science Foundation of China (NSFC) through the project [*Crossmodal Learning*](https://www.crossmodal-learning.org/home.html) (TRR-169).
