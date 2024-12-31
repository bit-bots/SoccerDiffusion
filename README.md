# DDLitLab2024 Project Hamburg Bit-Bots

End-to-end machine learning for soccer-playing robots. We are experimenting with diffusion-models to generate robot motions based on recent sensor measurements.

> [!IMPORTANT]
> This is still an ongioing research project.

> [!NOTE]
> This repository contains the source code for our [DDLitLab project](https://www.isa.uni-hamburg.de/ddlitlab.html) [Fußballspielende Roboter: Ende-zu-Ende-KI für Wahrnehmung und Steuerung im RoboCup](https://www.isa.uni-hamburg.de/ddlitlab/data-literacy-studierendenprojekte/vierte-foerderrunde/e2e-robot-soccer.html) for the fourth funding round.

## Getting Started

### Installation

> [!NOTE]
> The following installation steps are tested on Ubuntu 22.04 and 24.04.
> Please note, that these steps might fail on other systems.

1. Download this repo:

    ```shell
    git clone https://github.com/bit-bots/ddlitlab2024.git
    ```

2. Go into the downloaded directory:

    ```shell
    cd ddlitlab2024
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
