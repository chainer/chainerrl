from chainer import links as L


def copy_param(target_link, source_link):
    """Copy parameters of a link to another link."""
    target_params = dict(target_link.namedparams())
    for param_name, param in source_link.namedparams():
        if target_params[param_name].array is None:
            raise TypeError(
                'target_link parameter {} is None. Maybe the model params are '
                'not initialized.\nPlease try to forward dummy input '
                'beforehand to determine parameter shape of the model.'.format(
                    param_name))
        target_params[param_name].array[...] = param.array

    # Copy Batch Normalization's statistics
    target_links = dict(target_link.namedlinks())
    for link_name, link in source_link.namedlinks():
        if isinstance(link, L.BatchNormalization):
            target_bn = target_links[link_name]
            target_bn.avg_mean[...] = link.avg_mean
            target_bn.avg_var[...] = link.avg_var


def soft_copy_param(target_link, source_link, tau):
    """Soft-copy parameters of a link to another link."""
    target_params = dict(target_link.namedparams())
    for param_name, param in source_link.namedparams():
        if target_params[param_name].array is None:
            raise TypeError(
                'target_link parameter {} is None. Maybe the model params are '
                'not initialized.\nPlease try to forward dummy input '
                'beforehand to determine parameter shape of the model.'.format(
                    param_name))
        target_params[param_name].array[...] *= (1 - tau)
        target_params[param_name].array[...] += tau * param.array

    # Soft-copy Batch Normalization's statistics
    target_links = dict(target_link.namedlinks())
    for link_name, link in source_link.namedlinks():
        if isinstance(link, L.BatchNormalization):
            target_bn = target_links[link_name]
            target_bn.avg_mean[...] *= (1 - tau)
            target_bn.avg_mean[...] += tau * link.avg_mean
            target_bn.avg_var[...] *= (1 - tau)
            target_bn.avg_var[...] += tau * link.avg_var


def copy_grad(target_link, source_link):
    """Copy gradients of a link to another link."""
    target_params = dict(target_link.namedparams())
    for param_name, param in source_link.namedparams():
        if target_params[param_name].grad is None:
            if param.grad is None:
                pass
            else:
                target_params[param_name].grad = param.grad.copy()
        else:
            if param.grad is None:
                target_params[param_name].grad = None
            else:
                target_params[param_name].grad[...] = param.grad


def synchronize_parameters(src, dst, method, tau=None):
    {'hard': lambda: copy_param(dst, src),
     'soft': lambda: soft_copy_param(dst, src, tau),
     }[method]()
