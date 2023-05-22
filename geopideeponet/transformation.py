import numpy as np


class AffineTransformationSpace:
    r"""Affine transformation

    \phi(x) = A x + b

    A \in \mathbb{R}^{d \times d}
    b \in \mathbb{R}^d

    Args:
        d (int): dimension of x
        M (float): `M` > 0. The coefficients A, b are randomly sampled from [-`M`, `M`].
    """

    def __init__(self, d, M=1):
        self.d = d
        self.N = d**2 + d
        self.M = M

    def random(self, size):
        return 2 * self.M * np.random.rand(size, self.N) - self.M

    def eval_one(self, feature, x):
        A, b = self._get_a_b(feature)
        return A @ x + b

    @return_tensor
    def eval_batch(self, features, xs):
        mat = np.ones((len(features), len(xs), self.d))
        for i in range(len(features)):
            A, b = self._get_a_b(features[i])
            for j in range(len(xs)):
                mat[i, j] = A @ xs[j] + b
        return mat

    def _get_a_b(self, feature):
        A = np.reshape(feature[:self.d ** 2], (self.d, self.d))
        b = feature[-self.d:]
        return A, b


class PolarCoordinateTransformationSpace:
    r"""Polar coordinate transformation

    \phi(r, alpha) = ((r-r0) cos(alpha-alpha_0), (r-r0) sin(alpha-alpha_0))

    r_0, alpha_0 \in \mathbb{R}
    """
    def __init__(self):
        self.N = 2

    def random(self, size):
        return np.array([[1.0, 0.5]])
        # r = np.random.rand(size, 2)
        # r[:,1] *= 2 * np.pi
        # return r

    def eval_one(self, feature, x):
        r = x[0] + feature[0]
        alpha = x[1] * feature[1] * 2*np.pi
        return np.array((r * np.cos(alpha), r * np.sin(alpha)))

    @return_tensor
    def eval_batch(self, features, xs):
        assert features.shape[1] == xs.shape[1]
        mat = np.ones_like(features)
        for i in range(len(features)):
            mat[i] = features[i]
        return mat
