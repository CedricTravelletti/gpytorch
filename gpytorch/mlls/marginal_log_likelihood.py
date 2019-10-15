#!/usr/bin/env python3

from ..module import Module
from ..models import GP


class MarginalLogLikelihood(Module):
    """
    A module to compute the marginal log likelihood (MLL) of the GP model (or an approximate/bounded MLL)
    when applied to data.

    These models are typically used as the "loss" functions for GP models (though note that the output of
    these functions must be negated for optimization).

    Args:
        :attr:`likelihood` (:obj:`gpytorch.likelihoods.Likelihood`):
            The likelihood for the model
        :attr:`model` (:obj:`gpytorch.models.GP`):
            The approximate GP model
    """

    def __init__(self, likelihood, model):
        super(MarginalLogLikelihood, self).__init__()
        if not isinstance(model, GP):
            raise RuntimeError(
                "All MarginalLogLikelihood objects must be given a GP object as a model. If you are "
                "using a more complicated model involving a GP, pass the underlying GP object as the "
                "model, not a full PyTorch module."
            )
        self.likelihood = likelihood
        self.model = model

    def forward(self, output, target):
        """
        Computes the loss

        Args:
            :attr:`output` (:obj:`gpytorch.distributions.MultivariateNormal`):
                The outputs of the latent function (the :obj:`gpytorch.models.GP`)
            :attr:`target` (`torch.Tensor`):
                The target values (:math:`y`)
        """
        raise NotImplementedError
