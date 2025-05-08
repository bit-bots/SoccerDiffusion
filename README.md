# SoccerDiffusion

> Toward Learning End-to-End Humanoid Robot Soccer from Gameplay Recordings

Find our (preprint) paper and more information about the project on our [website](https://bit-bots.github.io/SoccerDiffusion/).

> [!IMPORTANT]
> This is still an ongoing research project.

## Getting Started

### Installation

> [!NOTE]
> The following installation steps are tested on Ubuntu 22.04 and 24.04.
> Please note, that these steps might fail on other systems.

1. Download this repo:

    ```shell
    git clone https://github.com/bit-bots/SoccerDiffusion.git
    ```

2. Go into the downloaded directory:

    ```shell
    cd soccer_diffusion
    ```

3. Install dependencies using [poetry](https://python-poetry.org/docs/#installation):

    ```shell
    poetry install --without test,dev
    ```

    Remove `test` or `dev` if you want to also install those optional dependencies.

4. Enter the poetry environment and run the code:

    ```shell
    poetry shell
    cli --help
    ```

### Optional Dependencies

Some tools contained in this repository require additional system-dependencies.

- `recording2mcap`: Requires a [ROS 2](https://docs.ros.org/en/jazzy/Installation.html) environment to work.
- `bhuman_importer`: Requires additional system dependencies to compile their Python-library for reading log files. (See [here](https://docs.b-human.de/master/getting-started/initial-setup/))

    ```shell
    sudo apt install ccache clang cmake libstdc++-12-dev llvm mold ninja-build
    ```

    Then build the Python package as described in [this document](https://docs.b-human.de/master/python-bindings/#local-build).

## Acknowledgements

We gratefully acknowledge funding and support from the project [*Digital and Data Literacy in Teaching Lab (DDLitLab)*](https://www.hcl.uni-hamburg.de/ddlitlab.html) at the University of Hamburg and the [*Stiftung Innovation in der Hochschullehre*](https://stiftung-hochschullehre.de/) foundation.
We extend our special thanks to the members of the [*Hamburg Bit-Bots*](https://bit-bots.de/) RoboCup team for their continuous support and for providing data and computational resources.
We also thank the RoboCup teams [*B-Human*](https://b-human.de/) and [*HULKs*](https://hulks.de/) for generously sharing their data for this research.
Additionally, we are grateful to the [*Technical Aspects of Multimodal Systems (TAMS)*](https://tams.informatik.uni-hamburg.de/) research group at the University of Hamburg for providing computational resources.
This research was partially funded by the Ministry of Science, Research and Equalities of Hamburg, as well as the German Research Foundation (DFG) and the National Science Foundation of China (NSFC) through the project [*Crossmodal Learning*](https://www.crossmodal-learning.org/home.html) (TRR-169).
