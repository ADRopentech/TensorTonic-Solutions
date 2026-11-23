import numpy as np
def gae(rewards: list, values: list, gamma: float, lam: float) -> list:
    """
    Returns the generalized advantage estimate at every timestep.
    """
    # Write code here
    sigma = []

    for t in range(len(rewards)):
        sigma.append(float(rewards[t] + gamma * values[t+1] - values[t]))

    A = [0.0] * len(sigma)
    A[-1] = sigma[-1]

    for t in range(len(sigma)-1, 0, -1):
        A[t-1] = float(sigma[t-1] + gamma * lam * A[t])

    return  A