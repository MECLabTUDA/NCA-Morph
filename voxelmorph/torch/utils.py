# import os
import random
import numpy as np
import torch
import SimpleITK as sitk
import pystrum.pynd.ndutils as nd

def set_all_seeds(seed):
  random.seed(seed)
  # os.environ("PYTHONHASHSEED") = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True

def rigid_align(fixed, moving, seg_m):
    r"""All inputs should we sitk Image objects not numpy arrays.
    """
    initial_transform = sitk.CenteredTransformInitializer(fixed, moving,
                                                          sitk.Euler3DTransform(),
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY
                                                         )
    moving = sitk.Resample(moving, fixed, initial_transform, sitk.sitkLinear, 0.0, moving.GetPixelID())
    # -- Use the transform between the images to apply it onto the segmentation as we don’t have GT during inference -- #
    seg_m = sitk.Resample(seg_m, fixed, initial_transform, sitk.sitkLinear, 0.0, seg_m.GetPixelID())
    return sitk.GetArrayFromImage(moving), sitk.GetArrayFromImage(seg_m)

def jacobian_determinant(disp):
    """
    https://github.com/MingR-Ma/RFR-WWANet/blob/main/tools.py#L74C1-L113C1
    jacobian determinant of a displacement field.
    NB: to compute the spatial gradients, we use np.gradient.

    Parameters:
        disp: 2D or 3D displacement field of size [*vol_shape, nb_dims],
              where vol_shape is of len nb_dims

    Returns:
        jacobian determinant (scalar)
    """

    volshape = disp.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

    grid_lst = nd.volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, len(volshape))

    J = np.gradient(disp + grid)

    if nb_dims == 3:
        dx = J[0]
        dy = J[1]
        dz = J[2]

        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

        return Jdet0 - Jdet1 + Jdet2

    else:

        dfdx = J[0]
        dfdy = J[1]

        return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]


def grid2contour(grid):
    '''
    https://github.com/MingR-Ma/RFR-WWANet/blob/main/tools.py#L114C1-L132C15
    grid--image_grid used to show deform field
    type: numpy ndarray, shape： (h, w, 2), value range：(-1, 1)
    '''
    x = np.arange(0, 96, 1)
    y = np.arange(0, 96, 1)
    X, Y = np.meshgrid(x, y)
    Z1 = grid[:, :, 0] + X

    Z2 = grid[:, :, 1] + Y

    fig = plt.figure()
    plt.contour(Z1, Y, X, 50, colors='darkgoldenrod')
    plt.contour(X, Z2, Y, 50, colors='darkgoldenrod')
    plt.xticks(()), plt.yticks(())
    plt.title('deform field')
    return fig

def get_nr_parameters(model):
    r"""This function returns the number of parameters and trainable parameters of a network.
        Based on: https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model
    """
    # -- Extract and count nr of parameters -- #
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # -- Return the information -- #
    return total_params, trainable_params

def get_model_size(model):
    r"""This function return the size in MB of a model.
        Based on: https://discuss.pytorch.org/t/finding-model-size/130275
    """
    # -- Extract parameter and buffer sizes -- #
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    # -- Transform into MB -- #
    size_all_mb = (param_size + buffer_size) / 1024**2
    # -- Return the size -- #
    return size_all_mb