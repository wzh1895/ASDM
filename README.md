# asdm

## **Agile System Dynamics Modelling**

ASDM is a python library that enables its users to create and simulate System Dynamics (SD) models. It can also simulate SD models saved in the XMILE format, including advanced features such as arrays and conveyors. The support is being continuously improved.

In the library:

- `asdm/asdm.py` consists of the main functionalities, including the lexer, parser, and interpreter.

- `asdm/utilities.py` provies a data visualisation tool.

- `asdm/Inference` consists of tools for model calibration.

## Installation
### Install from PyPi

```
pip install asdm
```
ASDM and its required dependencies will be automatically installed.

### Import

At any path, execute the following code in the interactive Python environment or as a part of a script:

```
from asdm import sdmodel
```

'sdmodel' is the class for System Dynamics models.

## Functionalities

Please refer to [Documentation](Documentation.md) for the commonly used functions.


## Tutorial Jupyter Notebooks

We also use Jupyter Notebooks to provide demoes of ASDM's functionalities.


[SD Modelling](demo/Demo_SD_modelling.ipynb)

- Creating an SD model from scratch
  - Adding stocks, flows, auxiliaries
  - Support for nonlinear functions (MIN, MAX, etc.)
  - Support for stochastic functions (random binomial trial, etc.)
- Running simulations
- Exporting and examing simulation outcomes
- Displaying simulation outcomes as graph

[Support for .stmx models](demo/Demo_stmx_support.ipynb)

- Load and simulate .stmx models
- Support for arrays
- Simulate .stmx models with modified equations

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