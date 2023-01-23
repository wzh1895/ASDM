import copy

class Var(object):
    """
    The 'value' data structure for SD variables, like Int or Float.
    It does not have a name, but can be linked to a name in the namespace.
    """
    def __init__(self, value=None, copy=None, dims=None):
        if copy is not None: 
            if  type(copy) is Var:
                if type(copy.value) is dict:
                    self.historical_values=dict()
                    self.value = dict()
                    for k, v in copy.value.items():
                        self.value[k] = 0
                        self.historical_values[k] = list()
                else:
                    self.historical_values = {'nosubscript':list()}
                    self.value = 0
            else:
                raise TypeError(type(copy))
        
        elif dims is not None:
            self.value = dict()
            self.historical_values = dict()
            for k in dims:
                self.value[k] = 0
                self.historical_values[k] = list()
        
        else:
            if type(value) is dict:
                self.value = value
                self.historical_values = dict()
                for k, _ in self.value.items():
                    self.historical_values[k] = list()
            elif type(value) is Var:
                self.value = copy.deepcopy(value.value)
                self.historical_values = copy.deepcopy(value.historical_values)
            else:
                self.value = float(value)
                self.historical_values = {'nosubscript':list()}
            
        # self.historical_values.append(value)

    def set(self, new_value):
        '''
        N.B. SVar = SubscriptedVar

        Var <- int/float OK
        SVar <- int/float NO
        Var <- Var OK
        SVar <- Var NO
        SVar <- SVar OK if dimensions match
        '''
        if type(self.value) is dict: # self is SVar
            if type(new_value) is Var and type(new_value.value) is dict: # SVar <- SVar
                if list(new_value.value.keys()) == list(self.value.keys()):
                    for k, v in new_value.value.items():
                        self.value[k] = v
                        self.historical_values[k].append(v)
                else:
                    raise Exception("Dimensions don't match.")
            else:
                raise Exception("Type of new value {} doesn't match".format(type(new_value)))
        else: # self is Var
            if type(new_value) is Var and type(new_value.value) is dict: #Var <- Svar
                raise Exception("Var<-Svar, NO")
            elif type(new_value) is Var and type(new_value.value) in [int, float]: #Var <- Var
                self.value = new_value.value 
                self.historical_values['nosubscript'].append(new_value.value)
            elif type(new_value) in [int, float]:
                self.value = new_value
                self.historical_values['nosubscript'].append(new_value)
            else:
                raise Exception("Unknown match issue: {}({}) and {}({})".format(type(self.value), self.value, type(new_value), new_value))

    def __add__(self, other):
        if type(other) in [int, float]:
            if type(self.value) in [int, float]:
                return Var(self.value + other)
            else:
                raise TypeError("Operator + cannot be used between {} ({}) and {} ({})".format(type(self), self, type(other), other))
        elif type(other) in [Var]:
            if type(self.value) in [int, float]:
                if type(other.value) in [int, float]:
                    return Var(self.value + other.value)
                else:
                    raise TypeError("Operator + cannot be used between {} ({}) and {} ({})".format(type(self), self, type(other), other))
            elif type(self.value) is dict:
                try:
                    new_value = dict()
                    for s, v in self.value.items():
                        new_value[s] = v - other.value[s]
                    return Var(value=new_value)
                except KeyError as e:
                    raise e
        else:
            raise TypeError("Operator + cannot be used between {} ({}) and {} ({})".format(type(self), self, type(other), other))

    def __radd__(self, other):
        if type(other) in [int, float]:
            if type(self.value) in [int, float]:
                return Var(self.value + other)
            else:
                raise TypeError("Operator + cannot be used between {} ({}) and {} ({})".format(type(other), other, type(self), self))
        elif type(other) in [Var]:
            if type(self.value) in [int, float]:
                if type(other.value) in [int, float]:
                    return Var(self.value + other.value)
                else:
                    raise TypeError("Operator + cannot be used between {} ({}) and {} ({})".format(type(other), other, type(self), self))
            elif type(self.value) is dict:
                try:
                    new_value = dict()
                    for s, v in self.value.items():
                        new_value[s] = v - other.value[s]
                    return Var(value=new_value)
                except KeyError as e:
                    raise e
        else:
            raise TypeError("Operator + cannot be used between {} ({}) and {} ({})".format(type(other), other, type(self), self))
    
    def __mul__(self, other):
        if type(self.value) is dict and (type(other) in [Var] and type(other.value) is dict):
            try:
                new_value = dict()
                for s, v in self.value.items():
                    new_value[s] = v * other.value[s]
                return Var(value=new_value)
            except KeyError as e:
                raise e
        
        elif type(self.value) is dict and type(other) in [int, float]:
            new_value = dict()
            for s, v in self.value.items():
                new_value[s] = v * other
            return Var(value=new_value)
            
        elif type(self.value) is dict and type(other.value) in [int, float]:
            new_value = dict()
            for s, v in self.value.items():
                new_value[s] = v * other.value
            return Var(value=new_value)
        elif type(self.value) in [int, float] and type(other) in [int, float]:
            return self.value * other

        elif type(self.value) in [int, float] and type(other.value) in [int, float]:
            return Var(self.value * other.value)
        else:
            raise TypeError("Operator * cannot be used between {} ({}) and {} ({})".format(type(self), self, type(other), other))

    def __rmul__(self, other):
        if type(self.value) is dict and (type(other) in [Var] and type(other.value) is dict):
            try:
                new_value = dict()
                for s, v in self.value.items():
                    new_value[s] = v * other.value[s]
                return Var(value=new_value)
            except KeyError as e:
                raise e
        elif type(self.value) is dict and type(other.value) in [int, float]:
            new_value = dict()
            for s, v in self.value.items():
                new_value[s] = v * other.value
            return Var(value=new_value)
        elif type(self.value) is dict and type(other) in [int, float]:
            new_value = dict()
            for s, v in self.value.items():
                new_value[s] = v * other
            return Var(value=new_value)
        elif type(self.value) in [int, float] and type(other) in [int, float]:
            return self.value * other

        elif type(self.value) in [int, float] and type(other.value) in [int, float]:
            return Var(self.value * other.value)
        else:
            raise TypeError("Operator * cannot be used between {} ({}) and {} ({})".format(type(other), other), type(self), self)

    def __sub__(self, other):
        if type(other) in [int, float]:
            if type(self.value) in [int, float]:
                return Var(self.value - other)
            else:
                raise TypeError("Operator - cannot be used between {} ({}) and {} ({})".format(type(self), self, type(other), other))
        elif type(other) in [Var]:
            if type(self.value) in [int, float]:
                if type(other.value) in [int, float]:
                    return Var(self.value - other.value)
                else:
                    new_value = dict()
                    for s, v in other.value.items():
                        new_value[s] = self.value - v
                    return Var(value=new_value)
            elif type(self.value) is dict:
                try:
                    new_value = dict()
                    for s, v in self.value.items():
                        new_value[s] = v - other.value[s]
                    return Var(value=new_value)
                except KeyError as e:
                    raise e
        else:
            raise TypeError("Operator - cannot be used between {} ({}) and {} ({})".format(type(self), self, type(other), other))
        
    # def __rsub__(self, other):
    #     if type(other) in [int, float]:
    #         if type(self.value) in [int, float]:
    #             return Var(other - self.value)
    #         elif type(self.value) is dict:
    #             if len(self.value) == 1: # only 1 dimension
    #                 return Var(other - self.value[next(iter(self.value))])
    #         else:
    #             raise TypeError("Operator - cannot be used between {} ({}) and {} ({})".format(type(self), self, type(other), other))
    #     elif type(other) in [Var]:
    #         if type(self.value) in [int, float]:
    #             if type(other.value) in [int, float]:
    #                 return Var(other.value - self.value)
    #             else:
    #                 new_value = dict()
    #                 for s, v in other.value.items():
    #                     new_value[s] = v - self.value
    #                 return Var(value=new_value)
    #         elif type(self.value) is dict:
    #             try:
    #                 new_value = dict()
    #                 for s, v in self.value.items():
    #                     new_value[s] = other.value[s] - v
    #                 return Var(value=new_value)
    #             except KeyError as e:
    #                 raise e
    #     else:
    #         raise TypeError("Operator - cannot be used between {} ({}) and {} ({})".format(type(self), self, type(other), other))

    def __truediv__(self, other):    

        if type(self.value) is dict and (type(other) in [Var] and type(other.value) is dict):
            try:
                new_value = dict()
                for s, v in self.value.items():
                    new_value[s] = v / other.value[s]
                return Var(value=new_value)
            except KeyError as e:
                raise e
        elif type(self.value) is dict and type(other.value) in [int, float]:
            new_value = dict()
            for s, v in self.value.items():
                new_value[s] = v / other.value
            return Var(value=new_value)
        elif type(self.value) is dict and type(other) in [int, float]:
            new_value = dict()
            for s, v in self.value.items():
                new_value[s] = v / other
            return Var(value=new_value)
        elif type(self.value) in [int, float] and type(other) in [int, float]:
            return Var(self.value / other)

        elif type(self.value) in [int, float] and type(other.value) in [int, float]:
            return Var(self.value / other.value)
        else:
            raise TypeError("Operator / cannot be used between {} ({}) and {} ({})".format(type(self), self, type(other), other))

    def __eq__(self, other):
        if self.value == other:
            return True
        else:
            return False
    
    def __gt__(self, other):
        pass

    def keys(self):
        return self.value.keys()

    def get_history(self, nsteps=None):
        if nsteps is None:
            return self.historical_values
        elif nsteps <= len(self.historical_values):
            return self.historical_values[-1*nsteps:]
        else:
            raise Exception("Arg nsteps out of range: {}".format(nsteps))

    def __repr__(self):
        if type(self.value) is dict:
            return 'SVar'+str(self.value)
        else:
            return str(self.value)

    def __setitem__(self, item, item_value):
        self.value[item] = item_value
        self.historical_values[item].append(item_value)

    def __getitem__(self, item):
        return self.value[item]
