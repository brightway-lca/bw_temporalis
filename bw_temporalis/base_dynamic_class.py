class TDAware:
    """Base class for functions which can be multiplied by temporal distributions"""

    def __mul__(self, other):
        raise NotImplementedError("Must be defined in child classes")
