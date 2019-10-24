#!/usr/bin/env python3

import torch
from 
from .. import settings
from .multi_device_kernel import MultiDeviceKernel


class ForwardedMultiDeviceKernel(MultiDeviceKernel):
    r"""
    Allocates the covariance matrix on distributed devices, e.g. multiple GPUs.
    VERSION FOR INVERSE PROBLEMS.

    Args:
        - :attr:`base_kernel`: Base kernel to distribute
        - :attr:`device_ids`: list of `torch.device` objects to place kernel chunks on
        - :attr:`output_device`: Device where outputs will be placed
    """

    def __init__(self, base_kernel, device_ids, F, output_device=None,
                 create_cuda_context=True, **kwargs):
        super(ForwardedMultiDeviceKernel, self). __init__(
            self, base_kernel, device_ids, F, output_device=None,
            create_cuda_context=True, **kwargs):

        self.F = F
    def forward(self, x1, x2, diag=False, F=None, **kwargs):
        print("In forwarded multidevice kernel forward.")
        if F is None:
            F = self.F
        if diag:
            pre_output = self.module.forward(x1, x2, diag=True, **kwargs).to(self.output_device)
            pre_output_F = pre_output.matmul(F)
            pre_output_F_t = pre_output_F._transpose_nonbatch()
            out = pre_output_F_t.matmul(F)
            return out

        if x1.size(-2) < len(self.device_ids) + 1:
            pre_output = self.module.forward(x1, x2, diag=diag, **kwargs).to(self.output_device)
            pre_output_F = pre_output.matmul(F)
            pre_output_F_t = pre_output_F._transpose_nonbatch()
            out = pre_output_F_t.matmul(F)
            return out

        if not x1.device == self.__cached_x1.device or not torch.equal(x1, self.__cached_x1):
            self._x1_scattered, self._kwargs = self.scatter((x1,), kwargs, self.device_ids)
            self.__cached_x1 = x1

        if not x2.device == self.__cached_x2.device or not torch.equal(x2, self.__cached_x2):
            self._x2_subs = [x2.to(x1_[0].device) for x1_ in self._x1_scattered]
            self.__cached_x2 = x2

        inputs = tuple((x1_[0], x2_) for x1_, x2_ in zip(self._x1_scattered, self._x2_subs))

        if not self.device_ids:
            pre_output = self.module.forward(*inputs, **self._kwargs)
            pre_output_F = pre_output.matmul(F)
            pre_output_F_t = pre_output_F._transpose_nonbatch()
            out = pre_output_F_t.matmul(F)
            return out

        if len(self.device_ids) == 1:
            pre_output = self.module.forward(*inputs[0], **self._kwargs[0])
            pre_output_F = pre_output.matmul(F)
            pre_output_F_t = pre_output_F._transpose_nonbatch()
            out = pre_output_F_t.matmul(F)

            return out

        # JIT modules can't be pickled and replicated yet
        # But reinitializing the distance_module every forward pass
        # is slow and should be removed once JIT modules can be pickled
        def set_distance_module_to_none(module):
            if hasattr(module, 'distance_module'):
                module.distance_module = None
        self.module.apply(set_distance_module_to_none)
        # Can't cache the replication because the base kernel module can change every time (e.g. param updates)
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])

        # TODO: parallel_apply might be too heavyweight in some cases?
        with settings.lazily_evaluate_kernels(False):
            outputs = self.parallel_apply(replicas, inputs, self._kwargs)

        pre_output = self.gather(outputs, self.output_device)
        pre_output_F = pre_output.matmul(F)
        pre_output_F_t = pre_output_F._transpose_nonbatch()

        out = pre_output_F_t.matmul(F)
        print("ForwardedMultiDeviceKernel forward.")
        print(out.shape)

        # Still have to multiply on other side.
        return out

    def gather(self, outputs, output_device):
        return CatLazyTensor(*[lazify(o) for o in outputs], dim=self.dim, output_device=self.output_device)

    def num_outputs_per_input(self, x1, x2):
        return self.base_kernel.num_outputs_per_input(x1, x2)

    # Override of the kernel one. Only if lazy evaluation is on, we want to
    # change to lazily evaluated forwarded kernel tensor.
    def __call__(self, x1, x2=None, diag=False, last_dim_is_batch=False, **params):
        x1_, x2_ = x1, x2

        # Delegate everything to Kernel class, we only want to override lazy
        # evaluation.
        if diag:
            res = super(ForwardedMultiDeviceKernel, self).__call__(x1_, x2_, diag=True, last_dim_is_batch=last_dim_is_batch, **params)

        else:
            if settings.lazily_evaluate_kernels.on():
                res = ForwardedLazyEvaluatedKernelTensor(
                        x1_, x2_, kernel=self,
                        last_dim_is_batch=last_dim_is_batch, **params)
            else:
                res = super(ForwardedMultiDeviceKernel, self).__call__(x1_, x2_, last_dim_is_batch=last_dim_is_batch, **params)
            return res
