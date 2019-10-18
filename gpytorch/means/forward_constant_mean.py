#!/usr/bin/env python3
""" Inmplements means of a forwarded GP.
I.e. mean is F * m_0 * 1_m, where m is model size.

Note that this implementation is awkward, since it takes as inputs model
locations and outputs data values.

(But this might be the only way to retain model covariance structure, i.e. to
make data inherit from model covariance.)

"""

import torch
from .mean import Mean
from ..utils.broadcasting import _mul_broadcast_shape
from ..utils.deprecation import _deprecate_kwarg_with_transform


class ForwardConstantMean(Mean):
    def __init__(self, prior=None, batch_shape=torch.Size(), **kwargs):
        batch_shape = _deprecate_kwarg_with_transform(
            kwargs, "batch_size", "batch_shape", batch_shape, lambda n: torch.Size([n])
        )
        super(ConstantMean, self).__init__()
        self.batch_shape = batch_shape
        self.register_parameter(name="constant", parameter=torch.nn.Parameter(torch.zeros(*batch_shape, 1)))
        if prior is not None:
            self.register_prior("mean_prior", prior, "constant")

    def forward(self, input, F):
        if input.shape[:-2] == self.batch_shape:
            return torch.mm(F, self.constant.expand(input.shape[:-1]))
        else:
            return self.constant.expand(_mul_broadcast_shape(input.shape[:-1], self.constant.shape))
