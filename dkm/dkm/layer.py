import math

import torch as t
import torch.nn as nn

from torch.utils import checkpoint

from .lib import StructMat
from .matrix_functions import chol_sqrt
from .util import pos_inv
from .output import Output

#Gives
def kl_reg(net):
    total = 0
    reg_layers = [layer for layer in net.modules() if
                  isinstance(layer, Layer) or isinstance(layer, Output)]
    for layer in reg_layers:
        total = total + layer.obj()
    return total

#
def norm_kl_reg(net, num_datapoints):
    return kl_reg(net) / num_datapoints


class Layer(nn.Module):
    def obj(self):
        return self.computed_obj

    def __init__(self, P, dof, sqrt=chol_sqrt, MAP=False):
        super().__init__()
        self.P = P
        self.dof = dof
        self.V = nn.Parameter(t.eye(self.P) * (
            math.sqrt(self.P)))  # In theory this should never be used since we change it in the first forward pass
        self.sqrt = sqrt
        self.MAP = MAP
        self.computed_obj = None
        self.inited = False
        self.latent_Gscale = nn.Parameter(t.zeros(()))

    # The "a" scaling here is pretty arbitrary. It could be changed.
    @property
    def G(self):
        G = self.V @ self.V.t() / self.V.shape[1]
        return G * t.exp(self.latent_Gscale)


class GramLayer(Layer):

    def compute_obj(self, K_inv, G):
        trace_term = t.trace(K_inv @ G)
        log_det_term = t.logdet(G) + t.logdet(K_inv)
        return -0.5 * self.dof * (-log_det_term + trace_term - G.shape[0])

    def _forward(self, *K_tuple):

        K = StructMat(*K_tuple)

        K_flat_ii = K.ii
        K_flat_ti = K.ti.reshape(-1, K.ti.shape[-1])
        K_flat_t = K.t.view(-1)
        K_flat = StructMat(K_flat_ii, K_flat_ti, K_flat_t)

        Kii_inv = pos_inv(K_flat.ii)
        Kti_inv_Kii_flat = K_flat.ti @ Kii_inv
        # ktt_i = K.t - t.diag(Kti_inv_Kii @ K.ti.t())
        ktt_i_flat = K_flat.t - t.sum(Kti_inv_Kii_flat * K_flat.ti, -1)

        Gii_flat = self.G
        Gti_flat = Kti_inv_Kii_flat @ Gii_flat
        gt_flat = t.sum(Gti_flat * Kti_inv_Kii_flat, -1) + ktt_i_flat
        assert (-1E-10 < gt_flat).all()

        Gii = Gii_flat.view(K.ii.shape)
        Gti = Gti_flat.reshape(K.ti.shape)
        gt = gt_flat.view(K.t.shape)

        # Compute and store objective
        computed_obj = self.compute_obj(Kii_inv, Gii)
        return Gii, Gti, gt, computed_obj

    def forward(self, K):
        # Initialize V
        if not self.inited:
            self.V.data = self.sqrt(K.ii) * (K.ii.shape[0]**0.5)
            self.inited = True

        Gii, Gti, gt, computed_obj = t.utils.checkpoint.checkpoint(self._forward, *K)

        self.computed_obj = computed_obj

        return StructMat(Gii, Gti, gt)


# Scale features by a matrix D, inheriting from Layer for convenience.
#TODO: Objective
class FeatureScaling(Layer):

    #Just put this here to make it clearer that this class needs N, the number of features, rather than P, the number of data points.
    def __init__(self, N, dof, sqrt=chol_sqrt, MAP=False):
        super().__init__(N, dof, sqrt, MAP)

    #In theory should be equivalent to the other compute_obj. This is the formula from the old "D Notation".
    def compute_obj(self):
        D = self.G
        trace = -0.5 * self.dof * (t.trace(D) - D.shape[-1])
        return trace


    def indiv_forward(self, X):

        Yp = X @ self.sqrt(self.G)
        return Yp

    def forward(self, Xs):
        if isinstance(Xs, tuple):
            X, Xt = Xs
            X_scaled, Xt_scaled = self.indiv_forward(X), self.indiv_forward(Xt)
            self.compute_obj()
            return (X_scaled, Xt_scaled)
        #TODO: Why is this else case here?
        else:
            assert isinstance(Xs, t.Tensor)
            return self.indiv_forward(Xs)
