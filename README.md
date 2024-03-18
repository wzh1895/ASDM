# asdm

## **Agile System Dynamics Modelling**

ASDM is a python library that enables its users to create and simulate System Dynamics (SD) models. It can also simulate SD models saved in the XMILE format, including advanced features such as arrays and conveyors. The support is being continuously improved.

In the library:

- `asdm/asdm.py` consists of the main functionalities, including the lexer, parser, and interpreter.

- `asdm/utilities.py` provies a data visualisation tool.

- `asdm/Inference` consists of tools for model calibration.

## Installation

### Python version

The library is developed and used with `Python 3.11`. However, it should also work with other versions. If you encounter a problem and believe it is related to python version, please open an issue or contact me. 

### Operating system

The library is developed and used on `macOS`. Some tests have also been done on `Windows` and `Ubuntu Linux`, but they are not comprehensive. If you encounter a problem and believe it is related to the OS, please open an issue or contact me.

### Create an environment

We recommend that you create a new environment to use the library, although this is not always necessary. For example, if you use anaconda, this can be done by:

```
conda create --name asdm
```

### Clone this repository to your local computer

To clone this repository to your local environment, please ensure that ```git``` is installed in your system, then use the following command:

```
git clone https://github.com/wzh1895/ASDM.git
```

### Install dependencies

ASDM relies on a number of other python libraries as dependencies. To install them, use the following commands:

```
cd asdm
conda install --file requirements.txt -c conda-forge
```

## Usage

Please refer to [Documentation](Documentation.md) for the commonly used functions.


## Tutorial Jupyter Notebooks

We also use Jupyter Notebooks to provide demoes of ASDM's functionalities.


1. [SD Modelling](demo/Demo_SD_modelling.ipynb)

- Creating an SD model from scratch
  - Adding stocks, flows, auxiliaries
  - Support for nonlinear functions (MIN, MAX, etc.)
  - Support for stochastic functions (random binomial trial, etc.)
- Running simulations
- Exporting and examing simulation outcomes
- Displaying simulation outcomes as graph

We will add more tutorial notebooks. 

You are welcomed to share your own tutorial notebooks through a `pull request`. When sharing notebooks, please make sure it does not contain sensitive data.

## Licence

ASDM is made public under the MIT licence.

## Contributors
**Wang Zhao** `main author`

PhD candidate & research assistant at University of Strathclyde, UK

Wang has given multiple talks on ASDM at different gatherings and conferences of modellers, operational researchers, and healthcare experts. This is the YouTube link to [one of the talks](https://www.youtube.com/watch?v=I_0YpIKc3yI&t=2321s).

Wang can be reached at <wang.zhao@strath.ac.uk>.

**Matt Stammers** `contributor`

Consultant Gastroenterologist & open-source software developer at University Hospital Southampton, UK

Matt created a webapp using streamlit to allow users to interact with a simulation dashboard in their web browsers or on smartphones, such as in this [demo](https://github.com/ReallyUsefulModels/Donanumab_Model). The simulation of the SD model in the backend is powered by ASDM. This is a part of an initiative [Really Useful Models](https://opendatasaveslives.org/news/2022-01-05-really-useful-models).

Matt's [GitHub Homepage](https://github.com/MattStammers)