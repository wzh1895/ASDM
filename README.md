# asdm

## **Agile System Dynamics Modelling**

This project is a Python-based tool that could be used for building, simulating, calibrating, and carrying out analyses of System Dynamics (SD) models.

## How to use

### Python version

The library is tested under Python 3.7.13 on Ubuntu Linux. It is supposed to work also on higher versions of Python. If you encounter problems using it please contact the author through email or open an issue.

### OS

It has been noticed that on macOS, the multi-thread sampling of PyMC3 does not work well. This is due to the fact that macOS handles multiprocess in a different way. The impact is that we can only use one thread to do multi-chain MCMC sampling which is much slower.

Although the rest of the functionalities of asdm still work on macOS, we recommend that you switch to Linux for the optimal parameter estimation experience.

### Dependencies

We recommend that you ues anaconda to manage the dependencies and use a dedicated fresh environment for asdm.

To create such an environment:

```
conda create --name asdm
```

To clone this repository to your local environment, please make sure that ```git``` is installed in your system and use the following command:

```
git clone https://github.com/wzh1895/asdm.git
```

To install all dependencies:

```
cd asdm
conda install --file requirements.txt -c conda-forge
```

## Tutorial Notebooks

We use a series of Jupyter Notebooks to provide functionality guidance for the users.

1. [SD Modelling](./1-SD_modelling.ipynb)

- Creating an SD model from scratch
  - Adding stocks, flows, auxiliaries
  - Support for nonlinear functions (MIN, MAX, etc.)
  - Support for stochastic functions (random binomial trial, etc.)
- Running simulations
- Exporting and examing simulation outcomes
- Displaying simulation outcomes as graph

2. [Model Calibration](./2-SD_model_calibration.ipynb)

- Estimating SD model parameters using Markov Chain Monte Carlo (MCMC)
  - Support for MLE (maximum likelihood) and MAP (maximum a posteriori) approach
  - Support for different modellings of payoff (squared error, Poisson, etc.) that are used in the likelihood function
  - Support for different sampling methods (Slice, Metropolis, NUTS, Hamiltonian, etc.), enabled by [PyMC3](https://docs.pymc.io/en/v3/)
- Diagnostic analyses of MCMC samples, enabled by [Arviz](https://arviz-devs.github.io/arviz/)
  - Trace plot
  - Autocorrelation plot
- Comparison between Historical data and simulation data

3. More to come...

## Why asdm?

Acknowledging that there are a number of SD tools that enable simulating SD models in a Python environment (see [PySD](https://github.com/JamesPHoughton/pysd) and [venpy](https://github.com/pbreach/venpy.git)), we build asdm to bring the capability of **sdmodel modelling** to the Python SD community. asdm allows users to create an SD model from scratch and edit model sdmodel as needed. With a native Python implementation of stock-and-flow sdmodel, asdm allows building SD models without software like [Vensim](https://vensim.com/) and [Stella](https://www.iseesystems.com/store/products/stella-architect.aspx).

The long-term goal asdm intends to achieve is an approach for **data-informed sdmodel modelling** where data can be used to finetune SD model sdmodel automatically. This is both an extension of data-informed SD model parameter estimation, and an innovative way to discover feedback sdmodels from data.

## Limitations

The main drawback at the moment is that we have not been working a lot on **translating Vensim/Stella models** into the format that asdm uses, although [PySD](https://github.com/JamesPHoughton/pysd) has proved this fully doable. This functionality is in our development roadmap.

We have neither been working a lot on **the speed of simulation**. Although simulating large SD models given mordern computer specs would not be extremely time-consuming, in tasks involing Monte Carlo simulation (such as parameter estimation) the speed of running a model can matter a lot more. Acceleration is in our development roadmap too.

## Licence

asdm is made public under the MIT licence.

## Author
**Wang Zhao**  
PhD candidate at University of Strathclyde, UK   
<wang.zhao@strath.ac.uk>  
