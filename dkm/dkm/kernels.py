import math
import torch as t
import torch.nn as nn

from .util import eye_like, pos_def
from .lib import StructMat, Weights

class IdentityKernel(nn.Module):
    def forward(self, K):
        return K

class Kernel(nn.Module):
    def __init__(self, mult_eps=1E-4, abs_eps=1E-4):
        super().__init__()
        self.mult_eps = mult_eps
        self.abs_eps = abs_eps

    def stabilise(self, Kii):
        I = eye_like(Kii)
        return Kii*(1.+self.mult_eps*I) + self.abs_eps*I

    def tuple_forward(self, *G_tuple):
        G = StructMat(*G_tuple)

        G_flat_ii = G.ii
        G_flat_ti = G.ti.reshape(-1,G.ti.shape[-1])
        G_flat_t = G.t.view(-1)
        G_flat = StructMat(G_flat_ii, G_flat_ti, G_flat_t)
        
        K = self._forward(G_flat)
        
        K.ii = self.stabilise(K.ii).view(G.ii.shape)
        K.ti = K.ti.reshape(G.ti.shape)
        K.t = (K.t*(1+self.mult_eps) + self.abs_eps).view(G.t.shape)

        return K.ii, K.ti, K.t

    def forward(self, G):
        return StructMat(*t.utils.checkpoint.checkpoint(self.tuple_forward, *G))
    
class StationaryKernel(Kernel):
    def R(self, G):
        g = G.diag()
        return -G + (0.5*g + 0.5*g[..., None])

    def _K(self, G):
        return self.k(self.R(G))

    def R_test(self, G):
        assert isinstance(G, StructMat)
        gi = G.ii.diag()
        return  -G.ti + (0.5*gi + 0.5*G.t[..., None])

    def _forward(self, G):
        assert isinstance(G, StructMat)
        Kii = self._K(G.ii)
        Kti = self.k(self.R_test(G))
        kt = self.k(t.zeros_like(G.t))
        return StructMat(Kii, Kti, kt)

class SqExpKernel(StationaryKernel):
    def k(self, R):
        return (-R).exp()

class ReluKernel(Kernel):
    eps=1E-6

    def __init__(self, bias_var = 0.1, weight_var=1, learned_bias=True, learned_weight=True, mult_eps=1E-4, abs_eps=1E-4):
        super().__init__(mult_eps=mult_eps, abs_eps=abs_eps)
        if bias_var <= 0 or weight_var <= 0:
            raise ValueError("Variance parameters must be positive")

        if learned_bias:
            self.log_bias_var = nn.Parameter(t.tensor(bias_var).log())
        else:
            self.register_buffer('log_bias_var', t.tensor(bias_var).log())
        if learned_weight:
            self.log_weight_var = nn.Parameter(t.tensor(weight_var).log())
        else:
            self.register_buffer('log_weight_var', t.tensor(weight_var).log())


    def component(self, ti, i, t):
        """
        Computes one matrix of covariances, not necessarily with diagonals

        The original expression:
        pi^{-1} ||x|| ||y|| (sin θ + (π - θ)cos θ)
        where
        cos θ = ti / √(i t)

        (1/π) √(i t)  (√(1 - ti²/(i t)) + (π - θ)ti / √(i t)
        which is equivalent to:
        (1/π) ( √(i t - ti²) + (π - θ) ti )

        In effect, inject noise along diagonal.
        """
        # input noise
        t_i = (i * (1 + self.mult_eps) + self.abs_eps) * (t[..., None] * (1 + self.mult_eps) + self.abs_eps)

        # Clamp these so the outputs are not NaN
        theta = (ti * t_i.rsqrt()).acos()
        t_i_sin_theta = (t_i - ti ** 2).sqrt()
        K = (t_i_sin_theta + (math.pi - theta) * ti) / math.pi

        if i is t:
            assert 2 == ti.dim()
            # Make sure the diagonal agrees with `i`
            Kv = K.view(*K.shape[:-2], -1)
            Kv[::(K.shape[-1] + 1)] = i * (1 + self.mult_eps) + self.abs_eps
            K = Kv.view(*K.shape)
            # K = K + self.epsilon * t.eye(K.shape[-1], device=K.device, dtype=K.dtype)
        return K

    def _K(self, G):
        assert pos_def(G, tol=1E-10)
        diag_ii = G.diagonal(dim1=-1, dim2=-2)
        K = self.component(G, diag_ii, diag_ii)
        assert pos_def(K)
        return K

    def _forward(self, G):
        diag_ii = G.ii.diagonal(dim1=-1, dim2=-2)
        ii = self.component(G.ii, diag_ii, diag_ii)*self.log_weight_var.exp() + self.log_bias_var.exp()
        ti = self.component(G.ti, diag_ii, G.t)*self.log_weight_var.exp() + self.log_bias_var.exp()
        return StructMat(ii, ti, G.t*self.log_weight_var.exp() + self.log_bias_var.exp())
    
class SumKernel(nn.Module):
    def __init__(self, kernels, weights, learned=False, lr_scale=1.):
        super().__init__()

        self.kernels = nn.ModuleList(kernels)
        self.weights = Weights(weights, learned, lr_scale)
        assert len(kernels) == len(weights)

    def forward(self, G):

        ws = nn.functional.softmax(self.weights(), dim=0)

        w0 = ws[0]
        K0 = self.kernels[0](G)
        Kii = w0*K0.ii
        Kti = w0*K0.ti
        Kt  = w0*K0.t
        
        for i in range(1, len(self.kernels)):
            wi = ws[i]
            Ki = self.kernels[i](G)
            Kii = Kii + wi*Ki.ii
            Kti = Kti + wi*Ki.ti
            Kt  = Kt  + wi*Ki.t

        return StructMat(Kii, Kti, Kt)