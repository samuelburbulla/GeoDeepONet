import numpy as np
import torch

# Unit square
def unit_square(n):
    h = 1 / (n - 1)
    xy = np.mgrid[0:1+1e-8:h, 0:1+1e-8:h].reshape(2, -1).T
    xy = torch.tensor(xy, requires_grad=True)
    return xy
