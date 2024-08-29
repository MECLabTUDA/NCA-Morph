import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from .. import default_unet_features
from . import layers
from .modelio import LoadableModel, store_config_args

class NCA_multires(LoadableModel):
    """
    A double CA architecture for segmentation on different resolutions.
    """
    @store_config_args
    def __init__(self, kernel_size = 7, steps = 30, fire_rate = 0.5, n_channels = 16, hidden_size = 64, steps_model2 = 30):
        r"""
        Parameters:
            kernel_size: Kernel size of NCA -> Relevant for perceptive field -> perceptive field = (kernel_size-1)/2 * steps
            steps: Times the NCA model will be applied to the input
            fire_rate = Chance that a cell is active at given step
            n_channels = Channels of NCA -> In channels are equal to out channels
            hidden_size = Hidden size of NCA
        """
        super().__init__()
        # -- Set variable that defines number of feature channels for NCAs output after forward pass -- #
        self.out_feats = n_channels # Set this dynamically

        # Model components
        self.fc0 = nn.Linear(n_channels*2, hidden_size)
        self.fc1 = nn.Linear(hidden_size, n_channels, bias=False)
        padding = int((kernel_size-1)/2)
        self.p0 = nn.Conv3d(n_channels, n_channels, kernel_size=kernel_size, stride=1, padding=padding, padding_mode="reflect")
        self.bn = torch.nn.BatchNorm3d(hidden_size)


        self.fc0_2 = nn.Linear(n_channels*2, hidden_size)
        self.fc1_2 = nn.Linear(hidden_size, n_channels, bias=False)
        self.p0_2 = nn.Conv3d(n_channels, n_channels, kernel_size=kernel_size, stride=1, padding=padding, padding_mode="reflect")
        self.steps_model2 = steps_model2

        self.avg_pool = torch.nn.AvgPool3d(9, 8, 3)
        self.avg_pool_4 = torch.nn.AvgPool3d(5, 4, 2)
        self.up_4 = torch.nn.Upsample(scale_factor=4, mode='nearest')
        self.up = torch.nn.Upsample(scale_factor=2, mode='nearest')
        # self.avg_pool = torch.nn.AvgPool3d(5, 4, 2)
        # self.up = torch.nn.Upsample(scale_factor=4, mode='nearest')

        # Model settings
        self.fire_rate = fire_rate
        self.steps = steps
        self.n_channels = n_channels

    def perceive(self, x):
        y = self.p0(x)
        y = torch.cat((x,y),1)
        return y

    def update(self, x_in):
        # x = x_in.transpose(1,4)
        # dx = self.perceive(x)
        dx = self.perceive(x_in)
        dx = dx.transpose(1,4)
        dx = self.fc0(dx)
        dx = dx.transpose(1,4)
        dx = self.bn(dx)
        dx = dx.transpose(1,4)
        dx = F.relu(dx)
        dx = self.fc1(dx)

        stochastic = torch.rand([dx.size(0),dx.size(1),dx.size(2), dx.size(3),1])<self.fire_rate
        stochastic = stochastic.float().cuda()
        dx = dx * stochastic
        x = x_in+dx.transpose(1,4)
        # x = x.transpose(1,4)

        return x

    def perceive2(self, x):
        y = self.p0_2(x)
        y = torch.cat((x,y),1)
        return y

    def update2(self, x_in):
        dx = self.perceive2(x_in)
        dx = dx.transpose(1,4)
        dx = self.fc0_2(dx)
        dx = dx.transpose(1,4)
        dx = self.bn(dx)
        dx = dx.transpose(1,4)
        dx = F.relu(dx)
        dx = self.fc1_2(dx)

        stochastic = torch.rand([dx.size(0),dx.size(1),dx.size(2), dx.size(3),1])<self.fire_rate
        stochastic = stochastic.float().cuda()
        dx = dx * stochastic
        x = x_in+dx.transpose(1,4)

        return x

    def forward(self, x):
        r"""
        Forward pass.
        """
        
        # Prepare input
        x_full = torch.zeros((x.shape[0], self.n_channels, x.shape[2], x.shape[3], x.shape[4]), dtype=torch.float32).cuda()
        x_full[:, 0:2, ...] = x
        x_downscaled = self.avg_pool(x_full)

        for step in range(self.steps):
            x_downscaled = self.update(x_downscaled)

        x_downscaled = self.up(x_downscaled)
        x_downscaled[:, 0:2, ...] = self.avg_pool_4(x_full)[:, 0:2, ...]

        for step in range(self.steps_model2):
            x_downscaled = self.update2(x_downscaled)

        x = self.up_4(x_downscaled)

        return x

class NCA_(LoadableModel):
    """
    A NCA architecture for segmentation.
    """
    @store_config_args
    def __init__(self, kernel_size = 7, steps = 30, fire_rate = 0.5, n_channels = 16, hidden_size = 64):
    # def __init__(self, kernel_size = 7, steps = 15, fire_rate = 0.5, n_channels = 16, hidden_size = 64):
    # def __init__(self, kernel_size = 5, steps = 3, fire_rate = 0.5, n_channels = 4, hidden_size = 16):
    # def __init__(self, kernel_size = 7, steps = 50, fire_rate = 0.5, n_channels = 16, hidden_size = 64):
    # def __init__(self, kernel_size = 7, steps = 30, fire_rate = 0.5, n_channels = 16, hidden_size = 64):
    # def __init__(self, kernel_size = 7, steps = 70, fire_rate = 0.5, n_channels = 16, hidden_size = 64):
    # def __init__(self, kernel_size = 7, steps = 50, fire_rate = 0.5, n_channels = 32, hidden_size = 64):
    # def __init__(self, kernel_size = 3, steps = 70, fire_rate = 0.75, n_channels = 16, hidden_size = 64):
        r"""
        Parameters:
            kernel_size: Kernel size of NCA -> Relevant for perceptive field -> perceptive field = (kernel_size-1)/2 * steps
            steps: Times the NCA model will be applied to the input
            fire_rate = Chance that a cell is active at given step
            n_channels = Channels of NCA -> In channels are equal to out channels
            hidden_size = Hidden size of NCA
        """
        super().__init__()
        # -- Set variable that defines number of feature channels for NCAs output after forward pass -- #
        self.out_feats = n_channels # Set this dynamically

        # Model components
        self.fc0 = nn.Linear(n_channels*2, hidden_size)
        self.fc1 = nn.Linear(hidden_size, n_channels, bias=False)
        padding = int((kernel_size-1)/2)
        self.p0 = nn.Conv3d(n_channels, n_channels, kernel_size=kernel_size, stride=1, padding=padding, padding_mode="reflect")
        self.bn = torch.nn.BatchNorm3d(hidden_size)
        
        # self.avg_pool = torch.nn.AvgPool3d(3, 2, 1)
        # self.up = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.avg_pool = torch.nn.AvgPool3d(3, 3, 0)
        self.up = torch.nn.Upsample(scale_factor=3, mode='nearest')
        # self.avg_pool = torch.nn.AvgPool3d(5, 4, 2)
        # self.up = torch.nn.Upsample(scale_factor=4, mode='nearest')

        # Model settings
        self.fire_rate = fire_rate
        self.steps = steps
        self.n_channels = n_channels

    def perceive(self, x):
        y = self.p0(x)
        y = torch.cat((x,y),1)
        return y

    def update(self, x_in):
        # x = x_in.transpose(1,4)
        # dx = self.perceive(x)
        dx = self.perceive(x_in)
        dx = dx.transpose(1,4)
        dx = self.fc0(dx)
        dx = dx.transpose(1,4)
        dx = self.bn(dx)
        dx = dx.transpose(1,4)
        dx = F.relu(dx)
        dx = self.fc1(dx)

        stochastic = torch.rand([dx.size(0),dx.size(1),dx.size(2), dx.size(3),1])<self.fire_rate
        stochastic = stochastic.float().cuda()
        dx = dx * stochastic
        x = x_in+dx.transpose(1,4)
        # x = x.transpose(1,4)

        return x

    def forward(self, x):
        r"""
        Forward pass.
        """
        # Prepare input
        x_full = torch.zeros((x.shape[0], self.n_channels, x.shape[2], x.shape[3], x.shape[4]), dtype=torch.float32).cuda()
        x_full[:, 0:2, ...] = x
        # x_downscaled = self.avg_pool(x_full)
        x_downscaled = x_full

        for step in range(self.steps):
            x_downscaled = self.update(x_downscaled)

        # x = self.up(x_downscaled)

        # if x_full.size() != x.size():

        #     # -- Zero pad to original size -- #
        #     x_ = torch.zeros((x.shape[0], x_full.size(1)-x.size(1), x.shape[2], x.shape[3], x.shape[4]), dtype=torch.float32).cuda()
        #     x = torch.concat([x, x_], dim=1)
        #     x_ = torch.zeros((x.shape[0], x.shape[1], x_full.size(2)-x.size(2), x.shape[3], x.shape[4]), dtype=torch.float32).cuda()
        #     x = torch.concat([x, x_], dim=2)
        #     x_ = torch.zeros((x.shape[0], x.shape[1], x.shape[2], x_full.size(3)-x.size(3), x.shape[4]), dtype=torch.float32).cuda()
        #     x = torch.concat([x, x_], dim=3)
        #     x_ = torch.zeros((x.shape[0], x.shape[1], x.shape[2], x.shape[3], x_full.size(4)-x.size(4)), dtype=torch.float32).cuda()
        #     x = torch.concat([x, x_], dim=4)

        # return x
        return x_downscaled

class UNet(LoadableModel):
    """
    A unet architecture for segmentation. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:

        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """
    @store_config_args
    def __init__(self,
                 inshape=None,
                 infeats=None,
                 nb_features=None,
                 nb_levels=None,
                 max_pool=2,
                 feat_mult=1,
                 feat_out=2,
                 nb_conv_per_level=1,
                 half_res=False):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            infeats: Number of input features.
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. 
                If None (default), the unet features are defined by the default config described in 
                the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. 
                Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
                Default is 1.
            nb_conv_per_level: Number of convolutions per unet level. Default is 1.
            half_res: Skip the last decoder upsampling. Default is False.
        """

        super().__init__()

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # cache some parameters
        self.half_res = half_res

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level)
            ]
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')

        # extract any surplus (full resolution) decoder convolutions
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        if isinstance(max_pool, int):
            max_pool = [max_pool] * self.nb_levels

        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        self.pooling = [MaxPooling(s) for s in max_pool]
        self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool]

        # configure encoder (down-sampling path)
        prev_nf = infeats
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.encoder.append(convs)
            encoder_nfs.append(prev_nf)

        # configure decoder (up-sampling path)
        encoder_nfs = np.flip(encoder_nfs)
        self.decoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.decoder.append(convs)
            if not half_res or level < (self.nb_levels - 2):
                prev_nf += encoder_nfs[level]

        # now we take care of any remaining convolutions
        self.remaining = nn.ModuleList()
        final_convs[-1] = feat_out
        for num, nf in enumerate(final_convs):
            self.remaining.append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf

        # cache final number of features
        self.final_nf = prev_nf

    def forward(self, x):

        # encoder forward pass
        x_history = [x]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                x = conv(x)
            x_history.append(x)
            x = self.pooling[level](x)

        # decoder forward pass with upsampling and concatenation
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x)
            if not self.half_res or level < (self.nb_levels - 2):
                x = self.upsampling[level](x)
                x = torch.cat([x, x_history.pop()], dim=1)

        # remaining convs at full resolution
        for conv in self.remaining:
            x = conv(x)

        return x

class Unet_(nn.Module):
    """
    A unet architecture. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:

        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """

    def __init__(self,
                 inshape=None,
                 infeats=None,
                 nb_features=None,
                 nb_levels=None,
                 max_pool=2,
                 feat_mult=1,
                 nb_conv_per_level=1,
                 half_res=False):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            infeats: Number of input features.
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. 
                If None (default), the unet features are defined by the default config described in 
                the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. 
                Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
                Default is 1.
            nb_conv_per_level: Number of convolutions per unet level. Default is 1.
            half_res: Skip the last decoder upsampling. Default is False.
        """

        super().__init__()

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # cache some parameters
        self.half_res = half_res

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level)
            ]
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')

        # extract any surplus (full resolution) decoder convolutions
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        if isinstance(max_pool, int):
            max_pool = [max_pool] * self.nb_levels

        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        self.pooling = [MaxPooling(s) for s in max_pool]
        self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool]

        # configure encoder (down-sampling path)
        prev_nf = infeats
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.encoder.append(convs)
            encoder_nfs.append(prev_nf)

        # configure decoder (up-sampling path)
        encoder_nfs = np.flip(encoder_nfs)
        self.decoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.decoder.append(convs)
            if not half_res or level < (self.nb_levels - 2):
                prev_nf += encoder_nfs[level]

        # now we take care of any remaining convolutions
        self.remaining = nn.ModuleList()
        for num, nf in enumerate(final_convs):
            self.remaining.append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf

        # cache final number of features
        self.final_nf = prev_nf

    def forward(self, x):

        # encoder forward pass
        x_history = [x]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                x = conv(x)
            x_history.append(x)
            x = self.pooling[level](x)

        # decoder forward pass with upsampling and concatenation
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x)
            if not self.half_res or level < (self.nb_levels - 2):
                x = self.upsampling[level](x)
                x = torch.cat([x, x_history.pop()], dim=1)

        # remaining convs at full resolution
        for conv in self.remaining:
            x = conv(x)

        return x

class Unet_down2(Unet_):
    """
    A unet architecture. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:

        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    
    Input gets downsampled by factor 3 and afterwards upsampled again
    """

    def __init__(self,
                 inshape=None,
                 infeats=None,
                 nb_features=None,
                 nb_levels=None,
                 max_pool=2,
                 feat_mult=1,
                 nb_conv_per_level=1,
                 half_res=False):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            infeats: Number of input features.
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. 
                If None (default), the unet features are defined by the default config described in 
                the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. 
                Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
                Default is 1.
            nb_conv_per_level: Number of convolutions per unet level. Default is 1.
            half_res: Skip the last decoder upsampling. Default is False.
        """

        super().__init__(inshape, infeats, nb_features, nb_levels, max_pool, feat_mult, nb_conv_per_level, half_res)

        self.avg_pool = torch.nn.AvgPool3d(3, 2, 1)
        self.up = torch.nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        x_s = x.size()
        x_ = self.avg_pool(x)
        x_ = super().forward(x_)
        x = self.up(x_)

        # -- Zero pad to original size -- #
        x_ = torch.zeros((x.shape[0], x.shape[1], x_s.size(2)-x.size(2), x.shape[3], x.shape[4]), dtype=torch.float32).cuda()
        x = torch.concat([x, x_], dim=2)
        x_ = torch.zeros((x.shape[0], x.shape[1], x.shape[2], x_s.size(3)-x.size(3), x.shape[4]), dtype=torch.float32).cuda()
        x = torch.concat([x, x_], dim=3)
        print(x.size(), x_s)
        raise
        return x

class VxmDense(LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    @store_config_args
    def __init__(self,
                 inshape,
                 nb_unet_features=None,
                 nb_unet_levels=None,
                 unet_feat_mult=1,
                 nb_unet_conv_per_level=1,
                 int_steps=7,
                 int_downsize=2,
                 bidir=False,
                 use_probs=False,
                 src_feats=1,
                 trg_feats=1,
                 unet_half_res=False):
        """ 
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. 
                If None (default), the unet features are defined by the default config described in 
                the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. 
                Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
                Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this 
                value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. 
                The flow field is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
            src_feats: Number of source image features. Default is 1.
            trg_feats: Number of target image features. Default is 1.
            unet_half_res: Skip the last unet decoder upsampling. Requires that int_downsize=2. 
                Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure core unet model
        self.unet_model = Unet_(
            inshape,
            infeats=(src_feats + trg_feats),
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
            half_res=unet_half_res,
        )

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.final_nf, ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError(
                'Flow variance has not been implemented in pytorch - set use_probs to False')

        # configure optional resize layers (downsize)
        if not unet_half_res and int_steps > 0 and int_downsize > 1:
            self.resize = layers.ResizeTransform(int_downsize, ndims)
        else:
            self.resize = None

        # resize to full res
        if int_steps > 0 and int_downsize > 1:
            self.fullsize = layers.ResizeTransform(1 / int_downsize, ndims)
        else:
            self.fullsize = None

        # configure bidirectional training
        self.bidir = bidir

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = layers.VecInt(down_shape, int_steps) if int_steps > 0 else None

        # configure transformer
        self.transformer = layers.SpatialTransformer(inshape)
        self.seg_transformer = layers.SpatialTransformer(inshape, mode='bilinear')

    def forward(self, source, target, seg, registration=False):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            seg: Source seg tensor.
            registration: Return transformed image and flow. Default is False.
        '''

        # concatenate inputs and propagate unet
        x = torch.cat([source, target], dim=1)
        # x = x.repeat(4, 1, 1, 1, 1)
        x = self.unet_model(x)

        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow

        # negate flow for bidirectional model
        neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow) if self.bidir else None

        # warp image with flow field as well as segmentation
        # source = source.repeat(4, 1, 1, 1, 1)
        y_source = self.transformer(source, pos_flow)
        seg = torch.nn.functional.one_hot(seg.squeeze(1).long()).permute(0, 4, 1, 2, 3).float()
        # seg = seg.repeat(4, 1, 1, 1, 1)
        y_seg = self.seg_transformer(seg, pos_flow)
        # y_seg[y_seg != 0] = 1
        # target = target.repeat(4, 1, 1, 1, 1)
        y_target = self.transformer(target, neg_flow) if self.bidir else None

        # return non-integrated flow field if training
        if not registration:
            return (y_source, y_target, preint_flow, y_seg) if self.bidir else (y_source, preint_flow, y_seg)
        else:
            return y_source, pos_flow, y_seg

class VxmNCA_direct_flow(LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images using NCA backbone extracting the flow directly from the NCA.
    """

    @store_config_args
    def __init__(self, inshape):
        """ 
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure core NCA model
        self.unet_model = NCA()

        # -- VxM variables that are used in forward but not for NCA version -- #
        self.resize = None
        self.fullsize = None
        self.bidir = False
        self.integrate = None

        # configure transformer
        self.transformer = layers.SpatialTransformer(inshape)
        self.seg_transformer = layers.SpatialTransformer(inshape, mode='bilinear')

    def forward(self, source, target, seg, registration=False):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            seg: Source seg tensor.
            registration: Return transformed image and flow. Default is False.
        '''

        # concatenate inputs and propagate unet
        x = torch.cat([source, target], dim=1)
        # x = x.repeat(4, 1, 1, 1, 1)
        x = self.unet_model(x)

        # transform into flow field
        flow_field = x[:, 2:5, ...]

        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow

        # negate flow for bidirectional model
        neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow) if self.bidir else None

        # warp image with flow field as well as segmentation
        # source = source.repeat(4, 1, 1, 1, 1)
        y_source = self.transformer(source, pos_flow)
        seg = torch.nn.functional.one_hot(seg.squeeze(1).long()).permute(0, 4, 1, 2, 3).float()
        # seg = seg.repeat(4, 1, 1, 1, 1)
        y_seg = self.seg_transformer(seg, pos_flow)
        # y_seg[y_seg != 0] = 1
        # target = target.repeat(4, 1, 1, 1, 1)
        y_target = self.transformer(target, neg_flow) if self.bidir else None

        # return non-integrated flow field if training
        if not registration:
            return (y_source, y_target, preint_flow, y_seg) if self.bidir else (y_source, preint_flow, y_seg)
        else:
            return y_source, pos_flow, y_seg

class VxmNCA_(LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images using NCA backbone using a specific conv layer for the flow.
    """

    @store_config_args
    def __init__(self, inshape):
        """ 
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure core NCA model
        self.unet_model = NCA()

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.out_feats, ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # -- VxM variables that are used in forward but not for NCA version -- #
        self.resize = None
        self.fullsize = None
        self.bidir = False
        self.integrate = None

        # configure transformer
        self.transformer = layers.SpatialTransformer(inshape)
        self.seg_transformer = layers.SpatialTransformer(inshape, mode='bilinear')

    def forward(self, source, target, seg, registration=False):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            seg: Source seg tensor.
            registration: Return transformed image and flow. Default is False.
        '''
        ret = VxmDense.forward(self, source, target, seg, registration)
        return ret

class NCA(LoadableModel):
    """
    A NCA architecture for segmentation.
    """
    @store_config_args
    def __init__(self, kernel_size = 7, steps = 30, fire_rate = 0.5, n_channels = 16, hidden_size = 64):
        r""" NCA shuld use patches as now downsampling is performed.
        Parameters:
            kernel_size: Kernel size of NCA -> Relevant for perceptive field -> perceptive field = (kernel_size-1)/2 * steps
            steps: Times the NCA model will be applied to the input
            fire_rate = Chance that a cell is active at given step
            n_channels = Channels of NCA -> In channels are equal to out channels
            hidden_size = Hidden size of NCA
        """
        super().__init__()
        # -- Set variable that defines number of feature channels for NCAs output after forward pass -- #
        self.out_feats = n_channels # Set this dynamically

        # Model components
        self.fc0 = nn.Linear(n_channels*2, hidden_size)
        self.fc1 = nn.Linear(hidden_size, n_channels, bias=False)
        padding = int((kernel_size-1)/2)
        self.p0 = nn.Conv3d(n_channels, n_channels, kernel_size=kernel_size, stride=1, padding=padding, padding_mode="reflect")
        self.bn = torch.nn.BatchNorm3d(hidden_size)
        
        # Model settings
        self.fire_rate = fire_rate
        self.steps = steps
        self.n_channels = n_channels

    def perceive(self, x):
        y = self.p0(x)
        y = torch.cat((x,y),1)
        return y

    def update(self, x_in):
        dx = self.perceive(x_in)
        dx = dx.transpose(1,4)
        dx = self.fc0(dx)
        dx = dx.transpose(1,4)
        dx = self.bn(dx)
        dx = dx.transpose(1,4)
        dx = F.relu(dx)
        dx = self.fc1(dx)

        stochastic = torch.rand([dx.size(0),dx.size(1),dx.size(2), dx.size(3),1])<self.fire_rate
        stochastic = stochastic.float().cuda()
        dx = dx * stochastic
        x = x_in+dx.transpose(1,4)

        return x

    def forward(self, x):
        r"""
        Forward pass.
        """
        # Prepare input
        x_full = torch.zeros((x.shape[0], self.n_channels, x.shape[2], x.shape[3], x.shape[4]), dtype=torch.float32).cuda()
        x_full[:, 0:2, ...] = x

        for step in range(self.steps):
            x_full = self.update(x_full)

        return x_full

class VxmNCA_db_cross(LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images using NCA backbone using a specific conv layer for the flow and patches.
    """

    @store_config_args
    def __init__(self, inshape, full_shape):
        """ 
        Parameters:
            inshape: Input shape. e.g. (32, 32, 32) --> patch shape
            full_shape: Full image shape. e.g. (192, 192, 192)
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure core NCA model
        self.unet_model = NCA_db_cross()

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.out_feats, ndims, kernel_size=3, padding=1)
        # -- Use this if NCA_db_cross is set and results are concatenated, not averaged! -- #
        # self.flow = Conv(self.unet_model.out_feats*2, ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # -- VxM variables that are used in forward but not for NCA version -- #
        self.resize = None
        self.fullsize = None
        self.bidir = False
        self.integrate = None

        # configure transformer
        self.transformer = layers.SpatialTransformer(inshape)
        self.seg_transformer = layers.SpatialTransformer(inshape, mode='bilinear')

        # configure transformer
        self.transformer_inf = layers.SpatialTransformer(full_shape)
        self.seg_transformer_inf = layers.SpatialTransformer(full_shape, mode='bilinear')

    def clamp(self, num, min_value, max_value):
        return int(max(min(num, max_value), min_value))

    def forward(self, source, target, seg, registration=False, inference=False):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            seg: Source seg tensor.
            registration: Return transformed image and flow. Default is False.
            inference: To set the correct grid, as during inference the full resolution image is used.
        '''
        ret = VxmDense.forward(self, source, target, seg, registration)
        # if not inference:
        #     # self.transformer = self.transformer_tr
        #     # self.seg_transformer = self.seg_transformer_tr
        #     ret = VxmDense.forward(self, source, target, seg, registration)
        # else:
        #     # self.transformer = self.transformer_inf
        #     # self.seg_transformer = self.seg_transformer_inf
        #     # ret = VxmDense.forward(self, source, target, seg, registration)

        #     # concatenate inputs and propagate unet
        #     x = torch.cat([source, target], dim=1)

        #     split_into = 4
        #     overlap = 0
        #     size_full = torch.tensor(x.shape[2:5])
        #     size = torch.floor(size_full/split_into)
        #     # -- If concat -- #
        #     # res = torch.zeros((x.shape[0], self.unet_model.n_channels*2, x.shape[2], x.shape[3], x.shape[4]), dtype=torch.float32).cuda()
        #     # -- If avg -- #
        #     res = torch.zeros((x.shape[0], self.unet_model.n_channels, x.shape[2], x.shape[3], x.shape[4]), dtype=torch.float32).cuda()
            
        #     for x_ in range(split_into):
        #         for y in range(split_into):
        #             for z in range(split_into):
        #                 start_x = self.clamp(int(size[0]*x_) - overlap, 0, size_full[0])
        #                 end_x = self.clamp(int(size[0]*(x_+1)) + overlap, 0, size_full[0])
        #                 start_y = self.clamp(int(size[1]*y) - overlap, 0, size_full[1])
        #                 end_y = self.clamp(int(size[1]*(y+1)) + overlap, 0, size_full[1])
        #                 start_z = self.clamp(int(size[2]*z) - overlap, 0, size_full[2])
        #                 end_z = self.clamp(int(size[2]*(z+1)) + overlap, 0, size_full[2])

        #                 res[:, :, start_x:end_x, start_y:end_y, start_z:end_z, ...] = self.unet_model(x[:, :, start_x:end_x, start_y:end_y, start_z:end_z, ...]).detach()
            
        #     x = res.clone()
        #     del res

        #     # transform into flow field
        #     flow_field = self.flow(x)

        #     # resize flow for integration
        #     pos_flow = flow_field
        #     if self.resize:
        #         pos_flow = self.resize(pos_flow)

        #     preint_flow = pos_flow

        #     # negate flow for bidirectional model
        #     neg_flow = -pos_flow if self.bidir else None

        #     # integrate to produce diffeomorphic warp
        #     if self.integrate:
        #         pos_flow = self.integrate(pos_flow)
        #         neg_flow = self.integrate(neg_flow) if self.bidir else None

        #         # resize to final resolution
        #         if self.fullsize:
        #             pos_flow = self.fullsize(pos_flow)
        #             neg_flow = self.fullsize(neg_flow) if self.bidir else None

        #     # warp image with flow field as well as segmentation
        #     y_source = self.transformer_inf(source, pos_flow)
        #     seg = torch.nn.functional.one_hot(seg.squeeze(1).long()).permute(0, 4, 1, 2, 3).float()
        #     y_seg = self.seg_transformer_inf(seg, pos_flow)
        #     y_target = self.transformer_inf(target, neg_flow) if self.bidir else None

        #     # return non-integrated flow field if training
        #     if not registration:
        #         ret = (y_source, y_target, preint_flow, y_seg) if self.bidir else (y_source, preint_flow, y_seg)
        #     else:
        #         ret = y_source, pos_flow, y_seg
            
        return ret

class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, 3, stride, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out

class NCA_db_cross(LoadableModel):
    """
    A double NCA architecture for registration interleaving knowledge between steps:
    """
    @store_config_args
    def __init__(self, kernel_size = 7, steps = 30, fire_rate = 0.5, n_channels = 16, hidden_size = 64):
        r"""
        Parameters:
            kernel_size: Kernel size of NCA -> Relevant for perceptive field -> perceptive field = (kernel_size-1)/2 * steps
            steps: Times the NCA model will be applied to the input
            fire_rate = Chance that a cell is active at given step
            n_channels = Channels of NCA -> In channels are equal to out channels
            hidden_size = Hidden size of NCA
        """
        super().__init__()
        # -- Set variable that defines number of feature channels for NCAs output after forward pass -- #
        self.out_feats = n_channels # Set this dynamically

        # Model components
        self.fc0 = nn.Linear(n_channels*2, hidden_size)
        self.fc1 = nn.Linear(hidden_size, n_channels, bias=False)
        padding = int((kernel_size-1)/2)
        self.p0 = nn.Conv3d(n_channels, n_channels, kernel_size=kernel_size, stride=1, padding=padding, padding_mode="reflect")
        self.bn = torch.nn.BatchNorm3d(hidden_size)

        self.fc0_2 = nn.Linear(n_channels*2, hidden_size)
        self.fc1_2 = nn.Linear(hidden_size, n_channels, bias=False)
        self.p0_2 = nn.Conv3d(n_channels, n_channels, kernel_size=kernel_size, stride=1, padding=padding, padding_mode="reflect")

        # Model settings
        self.fire_rate = fire_rate
        self.steps = steps
        self.n_channels = n_channels

        # Downsampling
        self.avg_pool = torch.nn.AvgPool3d(5, 4, 2)
        self.up = torch.nn.Upsample(scale_factor=4, mode='nearest')

    def perceive(self, x):
        y = self.p0(x)
        y = torch.cat((x,y),1)
        return y

    def update(self, x_in):
        dx = self.perceive(x_in)
        dx = dx.transpose(1,4)
        dx = self.fc0(dx)
        dx = dx.transpose(1,4)
        dx = self.bn(dx)
        dx = dx.transpose(1,4)
        dx = F.relu(dx)
        dx = self.fc1(dx)

        stochastic = torch.rand([dx.size(0),dx.size(1),dx.size(2), dx.size(3),1])<self.fire_rate
        stochastic = stochastic.float().cuda()
        dx = dx * stochastic
        x = x_in+dx.transpose(1,4)

        return x

    def perceive2(self, x):
        y = self.p0_2(x)
        y = torch.cat((x,y),1)
        return y

    def update2(self, x_in):
        dx = self.perceive2(x_in)
        dx = dx.transpose(1,4)
        dx = self.fc0_2(dx)
        dx = dx.transpose(1,4)
        dx = self.bn(dx)
        dx = dx.transpose(1,4)
        dx = F.relu(dx)
        dx = self.fc1_2(dx)

        stochastic = torch.rand([dx.size(0),dx.size(1),dx.size(2), dx.size(3),1])<self.fire_rate
        stochastic = stochastic.float().cuda()
        dx = dx * stochastic
        x = x_in+dx.transpose(1,4)

        return x

    def forward(self, x):
        r"""
        Forward pass. --> For every step between the two NCAs, knoweldge is interleaved.
        """
        target = x[:, 1, ...].unsqueeze(1)
        source = x[:, 0, ...].unsqueeze(1)

        # Prepare input
        x_fix = torch.zeros((target.shape[0], self.n_channels, target.shape[2], target.shape[3], target.shape[4]), dtype=torch.float32).cuda()
        x_fix[:, 0:2, ...] = target

        x_mov = torch.zeros((source.shape[0], self.n_channels, source.shape[2], source.shape[3], source.shape[4]), dtype=torch.float32).cuda()
        x_mov[:, 0:2, ...] = source

        x_fix = self.avg_pool(x_fix)
        x_mov = self.avg_pool(x_mov)

        for step in range(self.steps):
            x_fix = self.update(x_fix)
            x_mov = self.update2(x_mov)
            # -- Switch information between every step -- #
            x_fix_ = x_fix.clone()
            x_mov_ = x_mov.clone()
            x_fix[:, self.n_channels//2:, ...] = x_mov_[:, self.n_channels//2:, ...]
            x_mov[:, self.n_channels//2:, ...] = x_fix_[:, self.n_channels//2:, ...]
            del x_fix_, x_mov_
        
        x_fix = self.up(x_fix)
        x_mov = self.up(x_mov)

        # -- Here join the information again, either use concat or average the two results -- #
        # -- CONCAT -- #
        # res = torch.concat([x_fix, x_mov], dim=1)
        # -- AVERAGE -- #
        res = torch.mean(torch.stack([x_fix, x_mov]), dim=0)

        return res
    
class VxmNCA(LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images using NCA backbone using a specific conv layer for the flow and patches.
    """

    @store_config_args
    def __init__(self, inshape, full_shape):
        """ 
        Parameters:
            inshape: Input shape. e.g. (32, 32, 32) --> patch shape
            full_shape: Full image shape. e.g. (192, 192, 192)
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure core NCA model
        self.unet_model = NCA_db_unet()

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        # -- Use this if NCA_db_cross is set and results are concatenated, not averaged! -- #
        self.flow = Conv(self.unet_model.out_feats*2, ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # -- VxM variables that are used in forward but not for NCA version -- #
        self.resize = None
        self.fullsize = None
        self.bidir = False
        self.integrate = None

        # configure transformer
        self.transformer = layers.SpatialTransformer(inshape)
        self.seg_transformer = layers.SpatialTransformer(inshape, mode='bilinear')

        # configure transformer
        self.transformer_inf = layers.SpatialTransformer(full_shape)
        self.seg_transformer_inf = layers.SpatialTransformer(full_shape, mode='bilinear')

    def clamp(self, num, min_value, max_value):
        return int(max(min(num, max_value), min_value))

    def forward(self, source, target, seg, registration=False, inference=False):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            seg: Source seg tensor.
            registration: Return transformed image and flow. Default is False.
            inference: To set the correct grid, as during inference the full resolution image is used.
        '''
        ret = VxmDense.forward(self, source, target, seg, registration)
        # if not inference:
        #     # self.transformer = self.transformer_tr
        #     # self.seg_transformer = self.seg_transformer_tr
        #     ret = VxmDense.forward(self, source, target, seg, registration)
        # else:
        #     # self.transformer = self.transformer_inf
        #     # self.seg_transformer = self.seg_transformer_inf
        #     # ret = VxmDense.forward(self, source, target, seg, registration)

        #     # concatenate inputs and propagate unet
        #     x = torch.cat([source, target], dim=1)

        #     split_into = 4
        #     overlap = 0
        #     size_full = torch.tensor(x.shape[2:5])
        #     size = torch.floor(size_full/split_into)
        #     # -- If concat -- #
        #     # res = torch.zeros((x.shape[0], self.unet_model.n_channels*2, x.shape[2], x.shape[3], x.shape[4]), dtype=torch.float32).cuda()
        #     # -- If avg -- #
        #     res = torch.zeros((x.shape[0], self.unet_model.n_channels, x.shape[2], x.shape[3], x.shape[4]), dtype=torch.float32).cuda()
            
        #     for x_ in range(split_into):
        #         for y in range(split_into):
        #             for z in range(split_into):
        #                 start_x = self.clamp(int(size[0]*x_) - overlap, 0, size_full[0])
        #                 end_x = self.clamp(int(size[0]*(x_+1)) + overlap, 0, size_full[0])
        #                 start_y = self.clamp(int(size[1]*y) - overlap, 0, size_full[1])
        #                 end_y = self.clamp(int(size[1]*(y+1)) + overlap, 0, size_full[1])
        #                 start_z = self.clamp(int(size[2]*z) - overlap, 0, size_full[2])
        #                 end_z = self.clamp(int(size[2]*(z+1)) + overlap, 0, size_full[2])

        #                 res[:, :, start_x:end_x, start_y:end_y, start_z:end_z, ...] = self.unet_model(x[:, :, start_x:end_x, start_y:end_y, start_z:end_z, ...]).detach()
            
        #     x = res.clone()
        #     del res

        #     # transform into flow field
        #     flow_field = self.flow(x)

        #     # resize flow for integration
        #     pos_flow = flow_field
        #     if self.resize:
        #         pos_flow = self.resize(pos_flow)

        #     preint_flow = pos_flow

        #     # negate flow for bidirectional model
        #     neg_flow = -pos_flow if self.bidir else None

        #     # integrate to produce diffeomorphic warp
        #     if self.integrate:
        #         pos_flow = self.integrate(pos_flow)
        #         neg_flow = self.integrate(neg_flow) if self.bidir else None

        #         # resize to final resolution
        #         if self.fullsize:
        #             pos_flow = self.fullsize(pos_flow)
        #             neg_flow = self.fullsize(neg_flow) if self.bidir else None

        #     # warp image with flow field as well as segmentation
        #     y_source = self.transformer_inf(source, pos_flow)
        #     seg = torch.nn.functional.one_hot(seg.squeeze(1).long()).permute(0, 4, 1, 2, 3).float()
        #     y_seg = self.seg_transformer_inf(seg, pos_flow)
        #     y_target = self.transformer_inf(target, neg_flow) if self.bidir else None

        #     # return non-integrated flow field if training
        #     if not registration:
        #         ret = (y_source, y_target, preint_flow, y_seg) if self.bidir else (y_source, preint_flow, y_seg)
        #     else:
        #         ret = y_source, pos_flow, y_seg
            
        return ret

class NCA_db_unet(LoadableModel):
    """
    A double NCA architecture for registration interleaving knowledge between steps:
    """
    @store_config_args
    def __init__(self, kernel_size = 7, steps = 30, fire_rate = 0.5, n_channels = 16, hidden_size = 64):
        r"""
        Parameters:
            kernel_size: Kernel size of NCA -> Relevant for perceptive field -> perceptive field = (kernel_size-1)/2 * steps
            steps: Times the NCA model will be applied to the input
            fire_rate = Chance that a cell is active at given step
            n_channels = Channels of NCA -> In channels are equal to out channels
            hidden_size = Hidden size of NCA
        """
        super().__init__()
        # -- Set variable that defines number of feature channels for NCAs output after forward pass -- #
        self.out_feats = n_channels # Set this dynamically
        ndims = 3
        max_pool = [2, 2]
        self.encode1, self.encode2 = nn.ModuleList(), nn.ModuleList()
        self.encode1.append(ConvBlock(ndims, 1, 32))
        self.encode1.append(ConvBlock(ndims, 32, n_channels))
        self.encode2.append(ConvBlock(ndims, 1, 32))
        self.encode2.append(ConvBlock(ndims, 32, n_channels))

        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        self.pooling1 = [MaxPooling(s) for s in max_pool]
        self.pooling2 = [MaxPooling(s) for s in max_pool]


        # Model components
        self.fc0 = nn.Linear(n_channels*2, hidden_size)
        self.fc1 = nn.Linear(hidden_size, n_channels, bias=False)
        padding = int((kernel_size-1)/2)
        self.p0 = nn.Conv3d(n_channels, n_channels, kernel_size=kernel_size, stride=1, padding=padding, padding_mode="reflect")
        self.bn = torch.nn.BatchNorm3d(hidden_size)

        self.fc0_2 = nn.Linear(n_channels*2, hidden_size)
        self.fc1_2 = nn.Linear(hidden_size, n_channels, bias=False)
        self.p0_2 = nn.Conv3d(n_channels, n_channels, kernel_size=kernel_size, stride=1, padding=padding, padding_mode="reflect")

        # Model settings
        self.fire_rate = fire_rate
        self.steps = steps
        self.n_channels = n_channels

        # Downsampling
        # self.avg_pool = torch.nn.AvgPool3d(5, 4, 2)
        self.up = torch.nn.Upsample(scale_factor=4, mode='nearest')

    def perceive(self, x):
        y = self.p0(x)
        y = torch.cat((x,y),1)
        return y

    def update(self, x_in):
        dx = self.perceive(x_in)
        dx = dx.transpose(1,4)
        dx = self.fc0(dx)
        dx = dx.transpose(1,4)
        dx = self.bn(dx)
        dx = dx.transpose(1,4)
        dx = F.relu(dx)
        dx = self.fc1(dx)

        stochastic = torch.rand([dx.size(0),dx.size(1),dx.size(2), dx.size(3),1])<self.fire_rate
        stochastic = stochastic.float().cuda()
        dx = dx * stochastic
        x = x_in+dx.transpose(1,4)

        return x

    def perceive2(self, x):
        y = self.p0_2(x)
        y = torch.cat((x,y),1)
        return y

    def update2(self, x_in):
        dx = self.perceive2(x_in)
        dx = dx.transpose(1,4)
        dx = self.fc0_2(dx)
        dx = dx.transpose(1,4)
        dx = self.bn(dx)
        dx = dx.transpose(1,4)
        dx = F.relu(dx)
        dx = self.fc1_2(dx)

        stochastic = torch.rand([dx.size(0),dx.size(1),dx.size(2), dx.size(3),1])<self.fire_rate
        stochastic = stochastic.float().cuda()
        dx = dx * stochastic
        x = x_in+dx.transpose(1,4)

        return x

    def forward(self, x):
        r"""
        Forward pass. --> For every step between the two NCAs, knoweldge is interleaved.
        """
        target = x[:, 1, ...].unsqueeze(1)
        source = x[:, 0, ...].unsqueeze(1)

        # Conv pass
        # source_history = [source]
        for level, conv in enumerate(self.encode1):
            source = conv(source)
            # source_history.append(source)
            source = self.pooling1[level](source)

        # target_history = [target]
        for level, conv in enumerate(self.encode2):
            target = conv(target)
            # target_history.append(target)
            target = self.pooling2[level](target)

        # Prepare input
        x_fix = torch.zeros((target.shape[0], self.n_channels, target.shape[2], target.shape[3], target.shape[4]), dtype=torch.float32).cuda()
        x_fix[:, 0:self.n_channels, ...] = target

        x_mov = torch.zeros((source.shape[0], self.n_channels, source.shape[2], source.shape[3], source.shape[4]), dtype=torch.float32).cuda()
        x_mov[:, 0:self.n_channels, ...] = source

        # x_fix = self.avg_pool(x_fix)
        # x_mov = self.avg_pool(x_mov)

        for step in range(self.steps):
            x_fix = self.update(x_fix)
            x_mov = self.update2(x_mov)
            # -- Switch information between every step -- #
            # x_fix_ = x_fix.clone()
            # x_mov_ = x_mov.clone()
            # x_fix[:, self.n_channels//2:, ...] = x_mov_[:, self.n_channels//2:, ...]
            # x_mov[:, self.n_channels//2:, ...] = x_fix_[:, self.n_channels//2:, ...]
            # del x_fix_, x_mov_
        
        x_fix = self.up(x_fix)
        x_mov = self.up(x_mov)

        # -- Here join the information again, either use concat or average the two results -- #
        # -- CONCAT -- #
        res = torch.concat([x_fix, x_mov], dim=1)

        return res
    