class Var(object):
    """
    The 'value' data structure for SD variables, like Int of Float.
    It does not have a name, but can be linked to a name in namespace.
    """
    def __init__(self, value=None):
        self.value = value

    def __add__(self, other):
        if type(other) in [Var]:
            return Var(self.value + other.value)
        else:
            raise TypeError("Operator + cannot be used between {} and {}".format(type(self), type(other)))
    
    def __radd__(self, other):
        if type(other) in [Var]:
            return Var(self.value + other.value)
        else:
            raise TypeError("Operator + cannot be used between {} and {}".format(type(other), type(self)))

    def __repr__(self):
        return str(self.value)


class SD_AbstractSyntaxTree(object):
    def __init__(self):
        pass


if __name__ == "__main__":
    name_space = {
        'a': Var(10),
        'b': Var(20)
    }

    print(eval('a+b', name_space))
    print(eval('b+a', name_space))