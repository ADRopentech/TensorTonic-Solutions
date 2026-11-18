import numpy as np

def sigmoid(x: list | float) -> np.ndarray | float:
    """
    Returns the sigmoid value for a scalar or each element of a list.
    """
    # Write code here
    arr = np.asarray(x, dtype = float)
    sig_array = 1/ (1+ np.exp(-arr))

    if arr.ndim == 0 :
        return float(sig_array)
    return sig_array