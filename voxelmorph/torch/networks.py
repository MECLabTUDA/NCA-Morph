import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from voxelmorph.torch.model_utils import vitvnet_utils as v_utils
from voxelmorph.torch.model_utils import nicetrans_utils as n_utils
from voxelmorph.torch.model_utils import transmorph_utils as t_utils

from .. import default_unet_features
from . import layers
from .modelio import LoadableModel, store_config_args

class ViTVNet(LoadableModel):
    @store_config_args
    def __init__(self, img_size, int_steps=7, config=v_utils.get_3DReg_config(), vis=False):
        '''
        ViT-V-Net Model from https://github.com/junyuchen245/ViT-V-Net_for_3D_Image_Registration_Pytorch/blob/main/ViT-V-Net/models.py
        Added segmentation Transformation as well.
        '''
        super(ViTVNet, self).__init__()
        self.transformer = v_utils.Transformer(config, img_size, vis)
        self.decoder = v_utils.DecoderCup(config, img_size)
        self.reg_head = v_utils.RegistrationHead(
            in_channels=config.decoder_channels[-1],
            out_channels=config['n_dims'],
            kernel_size=3,
        )
        self.spatial_trans = v_utils.SpatialTransformer(img_size)
        self.config_ = config
        #self.integrate = VecInt(img_size, int_steps)

        # -- For segmentation warping -- #
        self.seg_transformer = v_utils.SpatialTransformer(img_size)


    def forward(self, source, target, seg, **kwargs):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            seg: Source seg tensor.
        '''
        # concatenate inputs and propagate unet
        x = torch.cat([source, target], dim=1)

        x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        x = self.decoder(x, features)
        flow = self.reg_head(x)
        #flow = self.integrate(flow)
        # -- Warp image and seg -- #
        y_source = self.spatial_trans(source, flow)
        seg = torch.nn.functional.one_hot(seg.squeeze(1).long()).permute(0, 4, 1, 2, 3).float()
        y_seg = self.seg_transformer(seg, flow)

        return y_source, flow, y_seg
    
class TransMorph(LoadableModel):
    @store_config_args
    # def __init__(self, img_size, config=t_utils.get_3DTransMorph_config()):
    def __init__(self, img_size, config=t_utils.get_3DTransMorphTiny_config()):
    # def __init__(self, img_size, config=t_utils.get_3DTransMorphSmall_config()):
    # def __init__(self, img_size, config=t_utils.get_3DTransMorphLarge_config()):
        '''
        TransMorph Model from https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/main/TransMorph/models/TransMorph.py
        Added segmentation Transformation as well.
        '''
        super(TransMorph, self).__init__()
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        self.transformer = t_utils.SwinTransformer(patch_size=config.patch_size,
                                           in_chans=config.in_chans,
                                           embed_dim=config.embed_dim,
                                           depths=config.depths,
                                           num_heads=config.num_heads,
                                           window_size=config.window_size,
                                           mlp_ratio=config.mlp_ratio,
                                           qkv_bias=config.qkv_bias,
                                           drop_rate=config.drop_rate,
                                           drop_path_rate=config.drop_path_rate,
                                           ape=config.ape,
                                           spe=config.spe,
                                           rpe=config.rpe,
                                           patch_norm=config.patch_norm,
                                           use_checkpoint=config.use_checkpoint,
                                           out_indices=config.out_indices,
                                           pat_merg_rf=config.pat_merg_rf,
                                           )
        self.up0 = t_utils.DecoderBlock(embed_dim*8, embed_dim*4, skip_channels=embed_dim*4 if if_transskip else 0, use_batchnorm=False)
        self.up1 = t_utils.DecoderBlock(embed_dim*4, embed_dim*2, skip_channels=embed_dim*2 if if_transskip else 0, use_batchnorm=False)  # 384, 20, 20, 64
        self.up2 = t_utils.DecoderBlock(embed_dim*2, embed_dim, skip_channels=embed_dim if if_transskip else 0, use_batchnorm=False)  # 384, 40, 40, 64
        self.up3 = t_utils.DecoderBlock(embed_dim, embed_dim//2, skip_channels=embed_dim//2 if if_convskip else 0, use_batchnorm=False)  # 384, 80, 80, 128
        self.up4 = t_utils.DecoderBlock(embed_dim//2, config.reg_head_chan, skip_channels=config.reg_head_chan if if_convskip else 0, use_batchnorm=False)  # 384, 160, 160, 256
        self.c1 = t_utils.Conv3dReLU(2, embed_dim//2, 3, 1, use_batchnorm=False)
        self.c2 = t_utils.Conv3dReLU(2, config.reg_head_chan, 3, 1, use_batchnorm=False)
        self.reg_head = t_utils.RegistrationHead(
            in_channels=config.reg_head_chan,
            out_channels=3,
            kernel_size=3,
        )
        self.spatial_trans = t_utils.SpatialTransformer(img_size)
        self.avg_pool = nn.AvgPool3d(3, stride=2, padding=1)
        # -- For segmentation warping -- #
        self.seg_transformer = t_utils.SpatialTransformer(img_size)

    def forward(self, source, target, seg, **kwargs):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            seg: Source seg tensor.
        '''

        # concatenate inputs and propagate unet
        x = torch.cat([source, target], dim=1)
        
        if self.if_convskip:
            x_s0 = x.clone()
            x_s1 = self.avg_pool(x)
            f4 = self.c1(x_s1)
            f5 = self.c2(x_s0)
        else:
            f4 = None
            f5 = None

        out_feats = self.transformer(x)

        if self.if_transskip:
            f1 = out_feats[-2]
            f2 = out_feats[-3]
            f3 = out_feats[-4]
        else:
            f1 = None
            f2 = None
            f3 = None
        x = self.up0(out_feats[-1], f1)
        x = self.up1(x, f2)
        x = self.up2(x, f3)
        x = self.up3(x, f4)
        x = self.up4(x, f5)
        flow = self.reg_head(x)
        # -- Warp image and seg -- #
        y_source = self.spatial_trans(source, flow)
        seg = torch.nn.functional.one_hot(seg.squeeze(1).long()).permute(0, 4, 1, 2, 3).float()
        y_seg = self.seg_transformer(seg, flow)

        return y_source, flow, y_seg
    
class NICE_Trans(LoadableModel):
    @store_config_args
    def __init__(self, 
                 in_channels=1, 
                 enc_channels=8, 
                 dec_channels=16, 
                 use_checkpoint=False): # --> use LoadableModel instead
        super(NICE_Trans, self).__init__()
        
        self.Encoder = n_utils.Conv_encoder(in_channels=in_channels,
                                    channel_num=enc_channels,
                                    use_checkpoint=use_checkpoint)
        self.Decoder = n_utils.Trans_decoder(in_channels=enc_channels,
                                     channel_num=dec_channels, 
                                     use_checkpoint=use_checkpoint)
        
        self.SpatialTransformer = n_utils.SpatialTransformer_block(mode='bilinear')
        # self.AffineTransformer = n_utils.AffineTransformer_block(mode='bilinear')

    def forward(self, source, target, seg, **kwargs):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            seg: Source seg tensor.
        '''

        x_fix = self.Encoder(target)
        x_mov = self.Encoder(source)
        flow, affine_para = self.Decoder(x_fix, x_mov)
        # -- Warp image and seg -- #
        y_source = self.SpatialTransformer(source, flow)
        seg = torch.nn.functional.one_hot(seg.squeeze(1).long()).permute(0, 4, 1, 2, 3).float()
        y_seg = self.SpatialTransformer(seg, flow)
        # affined = self.AffineTransformer(source, affine_para)
        
        return y_source, flow, y_seg

class NCA(LoadableModel):
    """
    A NCA architecture for segmentation.
    """
    @store_config_args
    # def __init__(self, kernel_size = 3, steps = 30, fire_rate = 0.5, n_channels = 16, hidden_size = 64):
    # def __init__(self, kernel_size = 5, steps = 30, fire_rate = 0.5, n_channels = 16, hidden_size = 64):
    # def __init__(self, kernel_size = 7, steps = 30, fire_rate = 0.5, n_channels = 16, hidden_size = 64):
    # def __init__(self, kernel_size = 9, steps = 30, fire_rate = 0.5, n_channels = 16, hidden_size = 64):
    # def __init__(self, kernel_size = 7, steps = 5, fire_rate = 0.5, n_channels = 16, hidden_size = 64):
    def __init__(self, kernel_size = 7, steps = 10, fire_rate = 0.5, n_channels = 16, hidden_size = 64):
    # def __init__(self, kernel_size = 7, steps = 50, fire_rate = 0.5, n_channels = 16, hidden_size = 64):
    # def __init__(self, kernel_size = 7, steps = 90, fire_rate = 0.5, n_channels = 16, hidden_size = 64):
    # def __init__(self, kernel_size = 7, steps = 10, fire_rate= 0.25, n_channels = 16, hidden_size = 64):
    # def __init__(self, kernel_size = 7, steps = 10, fire_rate = 0.5, n_channels = 16, hidden_size = 64):
    # def __init__(self, kernel_size = 7, steps = 10, fire_rate = 0.75, n_channels = 16, hidden_size = 64):
    # def __init__(self, kernel_size = 7, steps = 10, fire_rate = 1.0, n_channels = 16, hidden_size = 64):
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
        x_downscaled = self.avg_pool(x_full)
        # x_downscaled = x_full

        for step in range(self.steps):
            x_downscaled = self.update(x_downscaled)

        x = self.up(x_downscaled)

        if x_full.size() != x.size():

            # -- Zero pad to original size -- #
            x_ = torch.zeros((x.shape[0], x_full.size(1)-x.size(1), x.shape[2], x.shape[3], x.shape[4]), dtype=torch.float32).cuda()
            x = torch.concat([x, x_], dim=1)
            x_ = torch.zeros((x.shape[0], x.shape[1], x_full.size(2)-x.size(2), x.shape[3], x.shape[4]), dtype=torch.float32).cuda()
            x = torch.concat([x, x_], dim=2)
            x_ = torch.zeros((x.shape[0], x.shape[1], x.shape[2], x_full.size(3)-x.size(3), x.shape[4]), dtype=torch.float32).cuda()
            x = torch.concat([x, x_], dim=3)
            x_ = torch.zeros((x.shape[0], x.shape[1], x.shape[2], x.shape[3], x_full.size(4)-x.size(4)), dtype=torch.float32).cuda()
            x = torch.concat([x, x_], dim=4)

        return x
        # return x_downscaled

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

    def forward(self, source, target, seg, registration=True):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            seg: Source seg tensor.
            registration: Return transformed image and flow. Default is True.
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

class NCAMorph_direct_flow(LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images using NCA backbone extracting the flow directly from the NCA.
    """

    @store_config_args
    def __init__(self, inshape, kernel_size = 7, steps = 10, fire_rate = 0.5, n_channels = 16, hidden_size = 64, **kwargs):
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
        self.unet_model = NCA(kernel_size, steps, fire_rate, n_channels, hidden_size)

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

class NCAMorph(LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images using NCA backbone using a specific conv layer for the flow.
    """

    @store_config_args
    def __init__(self, inshape, kernel_size = 7, steps = 10, fire_rate = 0.5, n_channels = 16, hidden_size = 64, **kwargs):
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
        self.unet_model = NCA(kernel_size, steps, fire_rate, n_channels, hidden_size)

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
