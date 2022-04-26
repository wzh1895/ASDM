# Likelihood payoffs for a pair of (observation, simulation) data points at one time point.

import numpy as np
from scipy.stats import poisson


# poisson
# payoff contribution is always negative, see: https://www.vensim.com/documentation/payoffcomputation.html
def poisson_log_likelihood_payoff_contribution(observed, simulated, weight=1.0):

    if observed < 0:
        observed = 0  # data_n_events should be an integer >= 0

    if simulated <= 1:  # the modelled frequency of events must be at least 1
        simulated = 1

    log_likelihood = observed * np.log(simulated) - simulated
    return log_likelihood * weight


def poisson_log_likelihood(observed, simulated, weight=1.0):
    if observed < 0:
        observed = 0  # data_n_events should be an integer >= 0

    if simulated <= 1:  # the modelled frequency of events must be at least 1
        simulated = 1

    log_likelihood = poisson(simulated).logpmf(observed)
    # print('loglk', 'sim', simulated, 'obs', observed, 'loglk', log_likelihood)
    return log_likelihood * weight


def poisson_likelihood(observed, simulated, weight=1.0):

    if observed < 0:
        observed = 0  # data_n_events should be an integer >= 0

    if simulated <= 1:  # the modelled frequency of events must be at least 1
        simulated = 1

    likelihood = poisson(simulated).pmf(observed)

    # print('lk', 'sim', simulated, 'obs', observed, 'lk', likelihood)
    return likelihood * weight


# binomial
# payoff contribution is always negative, see: https://www.vensim.com/documentation/payoffcomputation.html
def binomial_log_likelihood_payoff_contribution(observed, simulated, p=0.5, weight=1.0):
    if observed < 0:
        observed = 0  # data_n_events should be an integer >= 0
    
    if simulated < 1:  # the modelled frequency of events must be an integer >= 0
        simulated = 1

    if observed > simulated:  # the observed frequency of events shoud not be greater than simulated (trails)
        observed = simulated

    return np.log((p**observed) * ((1-p)**(simulated - observed)))


# absolute error
# payoff contribution is always negative, see: https://www.vensim.com/documentation/payoffcomputation.html
def absolute_error_payoff_contribution(observed, simulated, weight=1.0):
    return (-1)*(((simulated-observed)*weight)**2)
