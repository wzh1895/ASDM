import numpy as np

class GraphFunc(object):
    def __init__(self, xscale, yscale, ypts):
        self.xscale = xscale
        self.yscale = yscale
        self.ypts = ypts
        self.eqn = None
        
        from scipy.interpolate import interp1d

        self.xpts = np.linspace(self.xscale[0], self.xscale[1], num=len(self.ypts))
        self.interp_func = interp1d(self.xpts, self.ypts, kind='linear')

    def __call__(self, input):
        # input out of xscale treatment:
        input = max(input, self.xscale[0])
        input = min(input, self.xscale[-1])
        output = float(self.interp_func(input)) # the output (like array([1.])) needs to be converted to float to avoid dimension explosion
        return output


class Conveyor(object):
    def __init__(self, length, eqn):
        self.length_time_units = length
        self.equation = eqn
        self.length_steps = None # to be decided at runtime
        self.initial_total = None # to be decided when initialising stocks
        self.conveyor = list() # pipe [new, ..., old]
    
    def initialize(self, length, value, leak_fraction=None):
        self.initial_total = value
        self.length_steps = length
        if leak_fraction is None or leak_fraction == 0:
            for _ in range(self.length_steps):
                self.conveyor.append(self.initial_total/self.length_steps)
        else:
            # print('Conveyor Initial Total:', self.initial_total)
            # print('Conveyor Leak fraction:', leak_fraction)
            # length_steps * output + nleaks * leak = initial_total
            # leak = output * (leak_fraction / length_in_steps)
            # ==> length_steps * output + n(length) * (output * (leak_fraction/length_in_steps)) = initial_total
            # print('Conveyor Length in steps:', self.length_steps)
            n_leak = 0
            for i in range(1, self.length_steps+1):
                n_leak += i
            # print('Conveyor N total leaks:', n_leak)
            output = self.initial_total / (self.length_steps + n_leak * (leak_fraction / self.length_steps))
            # print('Conveyor Output:', output)
            leak = output * (leak_fraction/self.length_steps)
            # print('Conveyor Leak:', leak)
            # generate slats
            for i in range(self.length_steps):
                self.conveyor.append(output + (i+1)*leak)
            self.conveyor.reverse()
        # print('Conveyor Initialised:', self.conveyor, '\n')

    def level(self):
        return sum(self.conveyor)

    def inflow(self, value):
        self.conveyor = [value] + self.conveyor

    def outflow(self):
        output = self.conveyor.pop()
        return output

    def leak_linear(self, leak_fraction):
        leaked = 0
        for i in range(self.length_steps):
            # print('Considering slat no.{}'.format(i+1))
            # keep slats non-negative
            if self.conveyor[i] > 0:
                already_leak_fraction = leak_fraction*((i+1)/self.length_steps) # 1+1: position indicator starting from 1
                # print('    Already leaked fraction:', already_leak_fraction)
                remaining_fraction = 1-already_leak_fraction
                original = self.conveyor[i] / remaining_fraction
                i_to_be_leaked = original * leak_fraction / self.length_steps
                # print('    Leaked no.{} to be leaked by {}:'.format(i+1, i_to_be_leaked))
                if self.conveyor[i] - i_to_be_leaked < 0:
                    i_to_be_leaked = self.conveyor[i]
                    self.conveyor[i] = 0
                else:
                    self.conveyor[i] = self.conveyor[i] - i_to_be_leaked
                leaked = leaked + i_to_be_leaked
            else:
                # print('    Slat no.{} is empty.'.format(i+1))
                pass
        # print('Leaked conveyor:', self.conveyor)
        # print('Leaked:', leaked)
        # print('\n')
        return leaked
