from logging import exception
from Payoff import poisson_log_likelihood_payoff_contribution, absolute_error_payoff_contribution
from MCMC import LogLike, LogLikeWithGrad
import pymc3 as pm
import theano.tensor as tt


class Optimizer(object):
    def __init__(
        self, 
        model, 
        parameters, 
        output_vars, 
        time_units, 
        dt=1, 
        payoff_type='Poisson', 
        inference_method='MLE', 
        ndraws=200, 
        nburn=800, 
        step_method='NUTS', 
        cores=1, 
        chains=1
        ):

        self.model = model
        self.parameters = parameters
        self.output_vars = output_vars
        self.time_units = time_units
        self.dt = dt
        
        if payoff_type == 'Squared error':
            self.payoff_function = absolute_error_payoff_contribution
        elif payoff_type == 'Poisson':
            self.payoff_function = poisson_log_likelihood_payoff_contribution
        else:
            raise Exception('Payoff function not specified')

        self.inference_method = inference_method
        self.ndraws = ndraws
        self.nburn = nburn

        self.step_method = step_method

        self.cores = cores
        self.chains = chains

    def calculate_payoff(self, params):
        self.model.clear_last_run()

        # set parameters
        for i in range(len(params)):
            p = list(self.parameters.keys())[i] # taking advantage of the ordered dict since Python 3.7
            self.model.replace_element_equation(p, params[i])

        # set stock initial values
        for ov, ts in self.output_vars.items():
            self.model.replace_element_equation(ov, ts[0])

        # simulate the sd model using parameters
        self.model.simulate(simulation_time=self.time_units, dt=self.dt)
        
        integral_payoff_over_simulation = 0

        for ov, ts in self.output_vars.items():
            sim = self.model.get_element_simulation_result(ov)
            for t in range(int(self.time_units/self.dt)):
                payoff_infected_t = self.payoff_function(ts[t], sim[t], weight=1.0/len(params))
                integral_payoff_over_simulation += payoff_infected_t * self.dt

        # whole payoff function
        if self.inference_method == 'MLE':
            payoff = integral_payoff_over_simulation
        elif self.inference_method == 'MAP':
            log_prior = query_joint_log_prior(inf, rt, log=True)
            payoff = log_prior + integral_payoff_over_simulation
        elif self.inference_method == 'prior':  # only sample the prior, no data include, for testing
            payoff = log_prior
        else:
            raise exception('Error: Estimate method {} not defined'.format(self.inference_method))

        # print('LL overall', log_posterior)
        # print(param, '\n',payoff, '\n')
        return payoff

    def run(self):
        # create likelihood Op
        # logl = LogLike(payoff_function)
        logl = LogLikeWithGrad(self.calculate_payoff)

        # use PyMC3 to sample from log-likelihood
        with pm.Model() as model:
            # set priors on theta
            params = list()
            for p, b in self.parameters.items():
                params.append(pm.Uniform(p, lower=b[0], upper=b[1]))

            theta = tt.as_tensor_variable(params)

            # create custom distribution
            # don't use DensityDist - use Potential. see: https://github.com/pymc-devs/pymc3/issues/4057#issuecomment-675589228
            pm.Potential('likelihood', logl(theta))

            # use a sampling method
            if self.step_method == 'Metropolis':
                self.step = pm.Metropolis()
            elif self.step_method == 'Slice':
                self.step = pm.Slice()
            elif self.step_method == 'NUTS':
                self.step = pm.NUTS()
            elif self.step_method == 'HamiltonianMC':
                self.step = pm.HamiltonianMC()
            else:
                print('Warning: Sampling method not specified. Falling back.')

        # use trace to collect all accepted samples
        with model:
            trace = pm.sample(self.ndraws,
                              tune=self.nburn,
                              discard_tuned_samples=True,
                              step=self.step,
                              cores=self.cores,
                              chains=self.chains,
                              return_inferencedata=False)

        return trace