import numpy as np

class GraphFunc(object):
    def __init__(self, yscale, ypts, xscale=None, xpts=None):
        self.yscale = yscale
        self.ypts = ypts
        self.eqn = None
        
        if xpts:
            self.xpts = xpts
        else:
            self.xscale = xscale
            self.xpts = np.linspace(self.xscale[0], self.xscale[1], num=len(self.ypts))
        
        from scipy.interpolate import interp1d

        self.interp_func = interp1d(self.xpts, self.ypts, kind='linear')

    def __call__(self, input):
        # input out of xscale treatment:
        input = max(input, self.xpts[0])
        input = min(input, self.xpts[-1])
        output = float(self.interp_func(input)) # the output (like array([1.])) needs to be converted to float to avoid dimension explosion
        return output


class Conveyor(object):
    def __init__(self, length, eqn):
        self.length_time_units = length
        self.equation = eqn
        self.length_steps = None # to be decided at runtime
        self.total = 0 # to be decided when initialising stocks
        self.slats = list() # pipe [new, ..., old]
        self.is_initialized = False
        self.leaks = list()
        self.leak_fraction = 0

    def initialize(self, length, value, leak_fraction=None):
        self.total = value
        self.length_steps = length
        if leak_fraction is None or leak_fraction == 0:
            for _ in range(self.length_steps):
                self.slats.append(self.total/self.length_steps)
                self.leaks.append(0)
        else:
            self.leak_fraction = leak_fraction
            n_leak = 0
            for i in range(self.length_steps):
                n_leak += i+1
            # print('Conveyor N total leaks:', n_leak)
            self.output = self.total / (self.length_steps + (n_leak * self.leak_fraction) / ((1-self.leak_fraction)*self.length_steps))
            # print('Conveyor Output:', output)
            leak = self.output * (self.leak_fraction/((1-self.leak_fraction)*self.length_steps))
            # print('Conveyor Leak:', leak)
            # generate slats
            for i in range(self.length_steps):
                self.slats.append(self.output + (i+1)*leak)
                self.leaks.append(leak)
            self.slats.reverse()
        # print('Conveyor Initialised:', self.conveyor, '\n')
        self.is_initialized = True

    def level(self):
        return self.total

    # order of execution:
    # 1 Leak from every slat
    #   to do this we need to know the leak for every slat
    # 2 Pop the last slat
    # 3 Input as the first slat

    def leak_linear(self):
        for i in range(self.length_steps):
            self.slats[i] = self.slats[i] - self.leaks[i]
        
        total_leaked = sum(self.leaks)
        self.total -= total_leaked
        return total_leaked
    
    def outflow(self):
        output = self.slats.pop()
        self.total -= output
        self.leaks.pop()
        return output

    def inflow(self, value):
        self.total += value
        self.slats = [value] + self.slats
        self.leaks = [value* self.leak_fraction/self.length_steps]+self.leaks
