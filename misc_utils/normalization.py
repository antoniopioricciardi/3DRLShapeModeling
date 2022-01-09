def normalize_vector(x, min_val=None, max_val=None):
    """
    normalize vector x to be in [-1,1] range
    x must be a numpy array.
    """
    if max_val is None or min_val is None:
        return (((x - min(x)) / (max(x) - min(x))) - 0.5) * 2
    return (((x - min_val) / (max_val - min_val)) - 0.5) * 2


def normalize_vector_positive(x, min_val=None, max_val=None):
    """
    normalize vector x to be in [0,1] range
    x must be a numpy array.
    """
    if max_val is None or min_val is None:
        return (x - min(x)) / (max(x) - min(x))
    return (x - min_val) / (max_val - min_val)

# import numpy as np
# a = np.array([1,2,3,4,5,6,7,8,9,10])
# print(normalize_vector(a))