# parts of the codes are from https://github.com/facebookresearch/preference-exploration/blob/main/test_functions.py

import torch
from botorch.test_functions.multi_objective import Penicillin
from torch import Tensor
################################ Multi-Objective Problem #########################
class VLMOP3:
   def __init__(self) -> None:
      self.bounds = torch.tensor([[-3.,-3.],[3.,3.]])

   def __call__(self, X: torch.Tensor) -> torch.Tensor:
      m = 3  # Assuming m is 3 based on the usage in the function
      X = X.reshape(-1,2)
      fX = torch.empty((m, X.shape[0]), dtype=X.dtype, device=X.device)

      fX[0, :] = 0.5 * (torch.square(X[:, 0]) + torch.square(X[:, 1])) + torch.sin(torch.square(X[:, 0]) + torch.square(X[:, 1]))
      fX[1, :] = torch.square(3*X[:, 0] - 2*X[:, 1] + 4) / 8 + torch.square(X[:, 0] - X[:, 1] + 1) / 27 + 15
      fX[2, :] = 1 / (torch.square(X[:, 0]) + torch.square(X[:, 1]) + 1) - 1.1 * torch.exp(-torch.square(X[:, 0]) - torch.square(X[:, 1]))

      return -fX.T

class positive_value_Penicillin(Penicillin):
    def evaluate_true(self, X: Tensor) -> Tensor:
        f = -super().evaluate_true(X)
        Y_bounds = torch.tensor(
            [
                [0., -86.93, - 514.70],
                [20, 0., 0.],
            ]
        ).to(X)
        f = (f - Y_bounds[0, :]) / (Y_bounds[1, :] - Y_bounds[0, :])
        return f
    
class CobbDouglas(torch.nn.Module):
    def __init__(self, theta,m=9):
        super().__init__()
        self.register_buffer("theta", theta)
        self.m = m
        
    def forward(self, Y, X=None):
            Y =Y.reshape(-1,self.m)
            result = 50*torch.prod((Y+0.5) ** self.theta,dim=1)  
            return result


class LinearExponential(torch.nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, Y, X=None):

        y0 = Y[..., 0] + 2
        y1 = Y[..., 1] + 2
       
        utility = 5*(y0 + y1) + 2*torch.exp(0.75*y0+1.25*y1)

        return utility

    

class LinearUtility(torch.nn.Module):
    def __init__(self, theta):
        super().__init__()
        self.register_buffer("theta", theta)

    def calc_raw_util_per_dim(self, Y):
        return Y * self.theta.to(Y)

    def forward(self, Y, X=None):
        return Y @ self.theta.to(Y)
    
    
    
class ExponentialUtility(torch.nn.Module):
    def __init__(self, theta,m=3):
        super().__init__()
        self.register_buffer("theta", theta)
        self.m = m


    def forward(self, Y, X=None):

          # m = 3
          theta_aux = torch.squeeze(self.theta)
          y_aux = Y.reshape(-1,self.m) #torch.squeeze(Y)
          val = torch.sum(1. - torch.exp(-theta_aux * y_aux), axis=1, keepdim=True) / theta_aux
          val /= self.m
          return val
      
      
class QuadraticUtility(torch.nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, Y, X=None):
        return torch.sum(Y**2, dim=-1)
      
#####################################################################################
import time
from copy import deepcopy
from typing import Dict, Optional

import numpy as np
import scipy
import torch
from torch.distributions import Kumaraswamy
from botorch.models.gp_regression import SingleTaskGP
from botorch.optim.utils import sample_all_priors
from botorch.test_functions.base import MultiObjectiveTestProblem
from botorch.test_functions.multi_objective import DTLZ2, VehicleSafety
from botorch.utils.gp_sampling import get_gp_samples
from gpytorch.constraints import GreaterThan
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
from gpytorch.priors import GammaPrior
from scipy.optimize import minimize
from torch import Tensor, square
from torch.quasirandom import SobolEngine



class AdaptedOSY(MultiObjectiveTestProblem):
    r"""
    Adapted OSY test problem from [Oszycka1995]_.
    This is adapted from botorch implementation.
    We negated the fs and treat gs a objectives so that the goal is to maximzie everything
    """

    dim = 6
    num_objectives = 8
    _bounds = [
        (0.0, 10.0),
        (0.0, 10.0),
        (1.0, 5.0),
        (0.0, 6.0),
        (1.0, 5.0),
        (0.0, 10.0),
    ]
    # Placeholder reference point
    _ref_point = [0.0] * 8

    def evaluate_true(self, X: Tensor) -> Tensor:
        f1 = (
            25 * (X[..., 0] - 2) ** 2
            + (X[..., 1] - 2) ** 2
            + (X[..., 2] - 1) ** 2
            + (X[..., 3] - 4) ** 2
            + (X[..., 4] - 1) ** 2
        )
        f2 = -(X ** 2).sum(-1)
        g1 = X[..., 0] + X[..., 1] - 2.0
        g2 = 6.0 - X[..., 0] - X[..., 1]
        g3 = 2.0 - X[..., 1] + X[..., 0]
        g4 = 2.0 - X[..., 0] + 3.0 * X[..., 1]
        g5 = 4.0 - (X[..., 2] - 3.0) ** 2 - X[..., 3]
        g6 = (X[..., 4] - 3.0) ** 2 + X[..., 5] - 4.0
        return torch.stack([f1, f2, g1, g2, g3, g4, g5, g6], dim=-1)/100


class NegativeVehicleSafety(VehicleSafety):
    def evaluate_true(self, X: Tensor) -> Tensor:
        f = -super().evaluate_true(X)
        Y_bounds = torch.tensor(
            [
                [-1.7040e03, -1.1708e01, -2.6192e-01],
                [-1.6619e03, -6.2136e00, -4.2879e-02],
            ]
        ).to(X)
        f = (f - Y_bounds[0, :]) / (Y_bounds[1, :] - Y_bounds[0, :])
        return f


class CarCabDesign(MultiObjectiveTestProblem):
    r"""RE9-7-1 car cab design from Tanabe & Ishibuchi (2020)"""

    dim = 7
    num_objectives = 9
    _bounds = [
        (0.5, 1.5),
        (0.45, 1.35),
        (0.5, 1.5),
        (0.5, 1.5),
        (0.875, 2.625),
        (0.4, 1.2),
        (0.4, 1.2),
    ]
    _ref_point = [0.0, 0.0]  # TODO: Determine proper reference point

    def evaluate_true(self, X: Tensor) -> Tensor:
        f = torch.empty(X.shape[:-1] + (self.num_objectives,), dtype=X.dtype, device=X.device)

        X1 = X[..., 0]
        X2 = X[..., 1]
        X3 = X[..., 2]
        X4 = X[..., 3]
        X5 = X[..., 4]
        X6 = X[..., 5]
        X7 = X[..., 6]
    

        # not using stochastic variables for the real function
        X8 = torch.zeros_like(X1)
        X9 = torch.zeros_like(X1)
        X10 = torch.zeros_like(X1)
        X11 = torch.zeros_like(X1)

        # First function
        # negate the first function as we want minimize car weight
        f[..., 0] = -(
            1.98
            + 4.9 * X1
            + 6.67 * X2
            + 6.98 * X3
            + 4.01 * X4
            + 1.75 * X5
            + 0.00001 * X6
            + 2.73 * X7
        )
        # Second function
        f[..., 1] = 1 - (
            1.16 - 0.3717 * X2 * X4 - 0.00931 * X2 * X10 - 0.484 * X3 * X9 + 0.01343 * X6 * X10
        )
        # Third function
        f[..., 2] = 0.32 - (
            0.261
            - 0.0159 * X1 * X2
            - 0.188 * X1 * X8
            - 0.019 * X2 * X7
            + 0.0144 * X3 * X5
            + 0.87570001 * X5 * X10
            + 0.08045 * X6 * X9
            + 0.00139 * X8 * X11
            + 0.00001575 * X10 * X11
        )
        # Fourth function
        f[..., 3] = 0.32 - (
            0.214
            + 0.00817 * X5
            - 0.131 * X1 * X8
            - 0.0704 * X1 * X9
            + 0.03099 * X2 * X6
            - 0.018 * X2 * X7
            + 0.0208 * X3 * X8
            + 0.121 * X3 * X9
            - 0.00364 * X5 * X6
            + 0.0007715 * X5 * X10
            - 0.0005354 * X6 * X10
            + 0.00121 * X8 * X11
            + 0.00184 * X9 * X10
            - 0.018 * X2 * X2
        )
        # Fifth function
        f[..., 4] = 0.32 - (
            0.74
            - 0.61 * X2
            - 0.163 * X3 * X8
            + 0.001232 * X3 * X10
            - 0.166 * X7 * X9
            + 0.227 * X2 * X2
        )
        # SiXth function
        tmp = (
            (
                28.98
                + 3.818 * X3
                - 4.2 * X1 * X2
                + 0.0207 * X5 * X10
                + 6.63 * X6 * X9
                - 7.77 * X7 * X8
                + 0.32 * X9 * X10
            )
            + (
                33.86
                + 2.95 * X3
                + 0.1792 * X10
                - 5.057 * X1 * X2
                - 11 * X2 * X8
                - 0.0215 * X5 * X10
                - 9.98 * X7 * X8
                + 22 * X8 * X9
            )
            + (46.36 - 9.9 * X2 - 12.9 * X1 * X8 + 0.1107 * X3 * X10)
        ) / 3
        f[..., 5] = 32 - tmp
        # Seventh function
        f[..., 6] = 32 - (
            4.72
            - 0.5 * X4
            - 0.19 * X2 * X3
            - 0.0122 * X4 * X10
            + 0.009325 * X6 * X10
            + 0.000191 * X11 * X11
        )
        # EighthEighth function
        f[..., 7] = 4 - (
            10.58
            - 0.674 * X1 * X2
            - 1.95 * X2 * X8
            + 0.02054 * X3 * X10
            - 0.0198 * X4 * X10
            + 0.028 * X6 * X10
        )
        # Ninth function
        f[..., 8] = 9.9 - (
            16.45
            - 0.489 * X3 * X7
            - 0.843 * X5 * X6
            + 0.0432 * X9 * X10
            - 0.0556 * X9 * X11
            - 0.000786 * X11 * X11
        )

        Y_bounds = torch.tensor(
            [
                [
                    -4.2150e01,
                    -4.7829e-01,
                    -1.1563e02,
                    -7.2040e-03,
                    -1.8255e-01,
                    -1.0168e01,
                    2.7023e01,
                    -8.0731e00,
                    -6.4556e00,
                ],
                # Old upper bound from 1e8 points
                # [-16.0992,   0.9511, 112.7138,   0.2750,   0.1909,  14.4804,  28.9855, -2.4875, -0.8270],
                # make upper bounds of constraints to be something > 0 so that it's possible to not violate the constraints
                [-16.0992, 0.9511, 112.7138, 0.2750, 0.1909, 14.4804, 28.9855, 0.5, 0.5],
            ]
        ).to(f)
        f = (f - Y_bounds[0, :]) / (Y_bounds[1, :] - Y_bounds[0, :])

        # normalize f to between 0 and 1 roughly so that we won't disadvantage ParEGO
        return f





class KumaraswamyCDF(torch.nn.Module):
    def __init__(self, concentration1, concentration2, Y_bounds):
        super().__init__()
        self.register_buffer("concentration1", concentration1)
        self.register_buffer("concentration2", concentration2)
        self.register_buffer("Y_bounds", Y_bounds)
        self.kdist = Kumaraswamy(concentration1, concentration2)

    def calc_raw_util_per_dim(self, Y):
        Y_bounds = self.Y_bounds

        Y = (Y - Y_bounds[0, :]) / (Y_bounds[1, :] - Y_bounds[0, :])
        eps = 1e-6
        Y = torch.clamp(Y, min=eps, max=1 - eps)

        util_val = self.kdist.cdf(Y)

        return util_val

    def forward(self, Y, X=None):
        util_val = self.calc_raw_util_per_dim(Y)
        util_val = util_val[..., ::2] * util_val[..., 1::2]
        util_val = util_val.sum(dim=-1)

        return util_val


class KumaraswamyCDFProduct(KumaraswamyCDF):
    def forward(self, Y, X=None):
        util_val = self.calc_raw_util_per_dim(Y)
        util_val = torch.prod(util_val, dim=-1)

        return util_val


