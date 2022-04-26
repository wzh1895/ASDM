import numpy as np

class PosteriorDistApprox(object):
    def __init__(self, observations, prior_func, likelihood_func):
        """
        observations: observations

        prior_func: function that yields the prior probability of x

        likelihood_func: function that yields the likelihood of y given x

        return: posterior probability of x

        """

        self.observations = observations
        self.samples = list()
        
        self.priors = list()
        self.likelihoods = list()
        self.joints=list()
        self.aggregated_joints = 0
        
        self.prior_func = prior_func
        self.likelihood_func = likelihood_func
        
        # self.posterior_trace = list()
    
    # consider new \theta_i, influencing the total probability
    def update(self, sample_points): # sample_points needs to be a list of points
        for sample_point in sample_points:
            log_prior = self.prior_func(sample_point)
            self.priors.append(log_prior)
        
            log_likelihood = self.likelihood_func(sample_point, self.observations)
            self.likelihoods.append(log_likelihood)
        
            joint = log_prior + log_likelihood
            self.joints.append(joint)
        
            self.aggregated_joints += joint # not sure if it's OK not to modify 'sum' to something else...
        
            # posterior = joint / self.aggregated_joints
            # self.posterior_trace.append(posterior)
            # print('pos', posterior)
        print("Bayesian update ready.")
        
    def get_posterior(self, sample_point):
        prior = self.prior_func(sample_point)
        likelihood = self.likelihood_func(sample_point, self.observations)
        # posterior = (prior+likelihood)/self.aggregated_joints
        posterior = (prior+likelihood) - self.aggregated_joints
        # print('post', posterior)
        return posterior


class PosteriorDist(object):
    def __init__(self, observations, prior_func, likelihood_func):
        """
        observations: observations

        prior_func: function that yields the prior probability of x

        likelihood_func: function that yields the likelihood of y given x

        return: posterior probability of x
        """

        self.observations = observations
        self.samples = list()
        
        self.priors = list()
        self.likelihoods = list()
        self.joints=list()
        self.aggregated_joints = 1
        
        self.prior_func = prior_func
        self.likelihood_func = likelihood_func
        
        # self.posterior_trace = list()
    
    # consider new \theta_i, influencing the total probability
    def update(self, sample_points): # sample_points needs to be a list of points
        for sample_point in sample_points:
            prior = self.prior_func(sample_point)
            self.priors.append(prior)
        
            likelihood = self.likelihood_func(sample_point, self.observations)
            self.likelihoods.append(likelihood)
        
            joint = prior * likelihood
            self.joints.append(joint)
        
            self.aggregated_joints *= joint # not sure if it's OK not to modify 'sum' to something else...
        
            # posterior = joint / self.aggregated_joints
            # self.posterior_trace.append(posterior)
            # print('pos', posterior)
        print("Bayesian update ready.")
        
    def get_posterior(self, sample_point):
        prior = self.prior_func(sample_point)
        likelihood = self.likelihood_func(sample_point, self.observations)
        posterior = (prior+likelihood)/self.aggregated_joints
        # print('post', posterior)
        return posterior    
