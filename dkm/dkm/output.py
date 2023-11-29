import torch as t
import torch.nn as nn

from torch.distributions import Normal
from torch.distributions import Categorical
from torch.distributions import MultivariateNormal

import torch.nn.functional as F

from .util import pos_inv

t.set_default_dtype(t.float64)


def gaussian_expectedloglikelihood(f_samples, y, obj_type="mean", noise_var=1):
    #If noise_var isn't a tensor, make one
    try:
        if not isinstance(noise_var, t.Tensor):
            noise_var = t.tensor([noise_var], device=f_samples.device)
    except:
        raise TypeError("noise_var must be a torch.Tensor / Parameter or a scalar")

    if obj_type == "sum":
        return Normal(f_samples.flatten(0, 1), noise_var.sqrt()).log_prob(y.repeat(f_samples.shape[0],1)).sum() / f_samples.shape[0]
    elif obj_type == "mean":
        #breakpoint()
        return Normal(f_samples.flatten(0,1), noise_var.sqrt()).log_prob(y.repeat(f_samples.shape[0],1)).mean(0).sum()
    else:
        raise ValueError("obj_type must be either 'sum' or 'mean'")


def gaussian_prediction(f_samples, noise_var=1):
    #If noise_var isn't a tensor, make one
    try:
        if not isinstance(noise_var, t.Tensor):
            noise_var = t.tensor([noise_var], device=f_samples.device)
    except:
        raise TypeError("noise_var must be a torch.Tensor / Parameter or a scalar")

    averaged_mean = f_samples.mean(0)
    averaged_var = f_samples.var(0) + noise_var
    return Normal(averaged_mean, averaged_var.sqrt())


def categorical_expectedloglikelihood(f_samples, y, obj_type="mean"):
    if obj_type == "sum":
        return -F.cross_entropy(f_samples.flatten(0, 1), y.repeat(f_samples.shape[0])) / f_samples.shape[0]
    elif obj_type == "mean":
        return -F.cross_entropy(f_samples.flatten(0, 1), y.repeat(f_samples.shape[0]))
    else:
        raise ValueError("obj_type must be either 'sum' or 'mean'")

def categorical_prediction(f_samples):
    averaged_log_prob = t.logsumexp(F.log_softmax(f_samples, dim=2), dim=0) - t.log(t.tensor([f_samples.shape[0]], device=f_samples.device))  # More numerically stable method of averaging probabilities over samples.
    return Categorical(t.exp(averaged_log_prob))


class Output(nn.Module):
    def __init__(self, P, out_features, mc_samples=1000):
        super().__init__()

        self.mu = nn.Parameter(t.randn(P, out_features))
        self.L  = nn.Parameter(t.randn(P, P))
        self.mc_samples = mc_samples

    #Shared A for all classes
    @property
    def A(self):
        return self.L @ self.L.t() / self.L.shape[-1]

    def compute_objs(self, K_inv, A, mus_mat):
        trace_term = t.trace(K_inv @ A)
        log_det_term = -(t.logdet(A) + t.logdet(K_inv))
        muSigmaQmu_term = t.sum((mus_mat.T @ K_inv) * mus_mat.T)
        return -0.5 * (1 / mus_mat.shape[1]) * (
                    (trace_term + log_det_term - A.shape[0]) * mus_mat.shape[1] + muSigmaQmu_term)

    def obj(self):
        return self.computed_obj

    def forward(self, K):
        Kii = K.ii
        Kti = K.ti
        Kt = K.t

        Kii_inv = pos_inv(Kii)
        Kti_inv_Kii = Kti @ Kii_inv
        mean_f = Kti_inv_Kii @ self.mu
        var_f = Kt - t.sum(Kti_inv_Kii * Kti, -1) + t.sum((Kti_inv_Kii @ self.A) * Kti_inv_Kii, -1)
        var_f = var_f[:, None]  # Since we share A for all output features, make sure var_f broadcasts correctly

        # Compute final regularisation term
        self.computed_obj = self.compute_objs(Kii_inv, self.A, self.mu)

        std_samples = t.randn((self.mc_samples, *mean_f.shape), device=mean_f.device)
        f_samples = std_samples * var_f.sqrt() + mean_f

        return f_samples