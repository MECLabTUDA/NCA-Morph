#!/usr/bin/env python

"""
Example script to register two volumes with VoxelMorph models.

Please make sure to use trained models appropriately. Let's say we have a model trained to register 
a scan (moving) to an atlas (fixed). To register a scan to the atlas and save the warp field, run:

    register.py --moving moving.nii.gz --fixed fixed.nii.gz --model model.pt 
        --moved moved.nii.gz --warp warp.nii.gz
        
The source and target input images are expected to be affinely registered.

If you use this code, please cite the following, and read function docs for further info/citations
    VoxelMorph: A Learning Framework for Deformable Medical Image Registration 
    G. Balakrishnan, A. Zhao, M. R. Sabuncu, J. Guttag, A.V. Dalca. 
    IEEE TMI: Transactions on Medical Imaging. 38(8). pp 1788-1800. 2019. 

    or

    Unsupervised Learning for Probabilistic Diffeomorphic Registration for Images and Surfaces
    A.V. Dalca, G. Balakrishnan, J. Guttag, M.R. Sabuncu. 
    MedIA: Medical Image Analysis. (57). pp 226-236, 2019 

Copyright 2020 Adrian V. Dalca

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in 
compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or 
implied. See the License for the specific language governing permissions and limitations under 
the License.
"""


# python register.py --moving /home/aranem_locale/Desktop/mnts/media/aranem_locale/AR_subs_exps/Lifelong-nnUNet-storage/MICCAI_2023/MICCAI_2023_raw_data/nnUNet_ready_with_ts_folder/Task110_RUNMC/imagesTr/Case08_0001.nii.gz --fixed /home/aranem_locale/Desktop/mnts/media/aranem_locale/AR_subs_exps/Lifelong-nnUNet-storage/MICCAI_2023/MICCAI_2023_raw_data/nnUNet_ready_with_ts_folder/Task110_RUNMC/imagesTr/Case08_0000.nii.gz --moving_seg /home/aranem_locale/Desktop/mnts/media/aranem_locale/AR_subs_exps/Lifelong-nnUNet-storage/MICCAI_2023/MICCAI_2023_raw_data/nnUNet_ready_with_ts_folder_BACKUP/Task110_RUNMC/labelsTr/Case08_0001.nii.gz --model /home/aranem_locale/Desktop/MICCAI_2023/experiments/vxm_torch_250_110_ncc_dice/0200.pt --out /home/aranem_locale/Desktop/MICCAI_2023/experiments/vxm_torch_250_110_ncc_dice/eval/Case08/ --gt /home/aranem_locale/Desktop/mnts/media/aranem_locale/AR_subs_exps/Lifelong-nnUNet-storage/MICCAI_2023/MICCAI_2023_raw_data/nnUNet_ready_with_ts_folder_BACKUP/Task110_RUNMC/labelsTr/Case08_0000.nii.gz


import os
import argparse
import pystrum
import torchio as tio
import SimpleITK as sitk

# third party
import torchio
import numpy as np
import torch
import shutil
from voxelmorph.torch.utils import set_all_seeds

# import voxelmorph with pytorch backend
os.environ['VXM_BACKEND'] = 'pytorch'
os.environ['NEURITE_BACKEND'] = 'pytorch'
import voxelmorph as vxm   # nopep8

# Set seeds for numpy, random and pytorch
set_all_seeds(3299)

def register():
    # parse commandline args
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, help='out path')
    parser.add_argument('--gt', required=True, help='GT segmentation path')
    parser.add_argument('--moving', required=True, help='moving image (source) filename')
    parser.add_argument('--fixed', required=True, help='fixed image (target) filename')
    parser.add_argument('--moving_seg', required=True, help='moving image (source) filename')
    parser.add_argument('--model', required=True, help='pytorch model for nonlinear registration')
    parser.add_argument('-g', '--gpu', help='GPU number(s) - if not supplied, CPU is used')
    parser.add_argument('--multichannel', action='store_true',
                        help='specify that data has multiple channels')
    parser.add_argument('--do_rigid', action='store_true',
                        help='specify moving and fixed image should be rigidly aligned first')
    parser.add_argument('--seg', action='store_true',
                        help='Set this to use VxM U-Net as segmentation network.')
    parser.add_argument('--nca', action='store_true',
                        help='Set this to use VxM NCA for registration.')
    parser.add_argument('--transmorph', action='store_true',
                        help='Set this to use TransMorph for registration.')
    parser.add_argument('--vitvnet', action='store_true',
                        help='Set this to use ViTVNet for registration.')
    parser.add_argument('--nicetrans', action='store_true',
                        help='Set this to use NICE-Trans for registration.')
    args = parser.parse_args()

    assert not(args.seg and args.nca), "Please only set one, either seg for segmentation with UNet or nca for registration but not both!"

    # device handling
    device = 'cuda'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # load moving and fixed images
    add_feat_axis = not args.multichannel
    moving = vxm.py.utils.load_volfile(args.moving, add_batch_axis=True, add_feat_axis=add_feat_axis)
    moving_seg = vxm.py.utils.load_volfile(args.moving_seg, add_batch_axis=True, add_feat_axis=add_feat_axis).astype(int)
    fixed, _ = vxm.py.utils.load_volfile(args.fixed, add_batch_axis=True, add_feat_axis=add_feat_axis, ret_affine=True)

    # load and set up model
    if args.nicetrans:
        model = vxm.networks.NICE_Trans.load(args.load_model, device)
    elif args.vitvnet:
        model = vxm.networks.ViTVNet.load(args.model, device)
    elif args.transmorph:
        model = vxm.networks.TransMorph.load(args.model, device)
    elif args.nca:
        model = vxm.networks.NCAMorph.load(args.model, device)
    elif args.seg:
        model = vxm.networks.UNet.load(args.model, device)
    else:
        model = vxm.networks.VxmDense.load(args.model, device)
    model.to(device)
    model.eval()

    # -- Print model size -- #
    # print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    # print(sum(p.numel() for p in model.parameters()))
    # raise

    # rigid alignment if desired
    if args.do_rigid and not args.seg:
        mov = sitk.GetImageFromArray(moving[0, ..., 0].transpose(2, 0, 1))
        mov_s = sitk.GetImageFromArray(moving_seg[0, ..., 0].transpose(2, 0, 1))
        fix = sitk.GetImageFromArray(fixed[0, ..., 0].transpose(2, 0, 1))
        moving, moving_seg = vxm.torch.utils.rigid_align(fix, mov, mov_s)
        # -- Transpose back to nibabel file order -- #
        moving = moving.transpose(1, 2, 0)[np.newaxis, ..., np.newaxis]
        moving_seg = moving_seg.transpose(1, 2, 0)[np.newaxis, ..., np.newaxis]

    # set up tensors and permute
    input_moving = torch.from_numpy(moving).to(device).float().permute(0, 4, 1, 2, 3)
    input_seg = torch.from_numpy(moving_seg).to(device).float().permute(0, 4, 1, 2, 3)
    input_fixed = torch.from_numpy(fixed).to(device).float().permute(0, 4, 1, 2, 3)
    fixed_seg = vxm.py.utils.load_volfile(args.gt, add_batch_axis=True, add_feat_axis=add_feat_axis).astype(int)
    fixed_seg = torch.from_numpy(fixed_seg).to(device).float().permute(0, 4, 1, 2, 3)




    # # -- Bring to correct size -- #
    # resize = torchio.transforms.Resize(target_shape=(160, 224, 192))
    # input_moving = resize(input_moving[0].cpu()).unsqueeze(0).cuda()
    # input_seg = resize(input_seg[0].cpu()).unsqueeze(0).cuda()
    # input_fixed = resize(input_fixed[0].cpu()).unsqueeze(0).cuda()
    # fixed_seg = resize(fixed_seg[0].cpu()).unsqueeze(0).cuda()
    # -- Uncomment to bring input to correct size for brain -- #




    

    # -- Do augmentations here -- #
    # x = 2   # Level 1
    # x = 4   # Level 2
    # x = 6   # Level 3
    
    # bias_4 = torchio.transforms.RandomBiasField(coefficients=0.3*x)
    # input_moving = bias_4(input_moving[0].cpu()).unsqueeze(0).cuda() # Must be 4D tensor
    
    # ghosting_4 = torchio.transforms.RandomGhosting(num_ghosts=x, intensity=0.25*x)
    # input_moving = ghosting_4(input_moving[0]).unsqueeze(0) # Must be 4D tensor
    
    # spike_4 = torchio.transforms.RandomSpike(num_spikes=x, intensity=0.25*x)
    # input_moving = spike_4(input_moving[0]).unsqueeze(0) # Must be 4D tensor
    
    # noise_4 = torchio.transforms.RandomNoise(mean=0.05*x, std=0.025*x)
    # input_moving = noise_4(input_moving[0].cpu()).unsqueeze(0).cuda() # Must be 4D tensor
    
    # anisotropy_4 = torchio.transforms.RandomAnisotropy(downsampling=x)
    # input_moving = anisotropy_4(input_moving[0].cpu()).unsqueeze(0).cuda() # Must be 4D tensor

    # flip = torchio.transforms.RandomFlip(axes=('left',), flip_probability=1)
    # input_moving = flip(input_moving[0].cpu()).unsqueeze(0).cuda() # Must be 4D tensor





    # -- Shift images and labels n pixels -- #
    
    # -- Shift to the left -- #
    # n = -50
    # flip = torchio.transforms.RandomFlip(axes=('left',), flip_probability=1)
    # input_moving = torch.roll(input_moving, shifts=-n, dims=2)  # 4
    # input_moving[:,:,:-n,:] = 0
    # input_fixed = torch.roll(input_fixed, shifts=-n, dims=2)  # 4
    # input_fixed[:,:,:-n,:] = 0
    # input_seg = torch.roll(input_seg, shifts=-n, dims=2)  # 4
    # input_seg[:,:,:-n,:] = 0
    # input_seg = flip(input_seg[0].cpu()).unsqueeze(0).cuda()
    # fixed_seg = torch.roll(fixed_seg, shifts=-n, dims=2)  # 4
    # fixed_seg[:,:,:-n,:] = 0
    # fixed_seg = flip(fixed_seg[0].cpu()).unsqueeze(0).cuda()

    # -- Shift to the bottom -- #
    # n = -50
    # flip = torchio.transforms.RandomFlip(axes=('left',), flip_probability=1)
    # input_moving = torch.roll(input_moving, shifts=-n, dims=3)  # 4
    # input_moving[:,:,:,:-n, :] = 0
    # input_fixed = torch.roll(input_fixed, shifts=-n, dims=3)  # 4
    # input_fixed[:,:,:,:-n, :] = 0
    # input_seg = torch.roll(input_seg, shifts=-n, dims=3)  # 4
    # input_seg[:,:,:,:-n, :] = 0
    # input_seg = flip(input_seg[0].cpu()).unsqueeze(0).cuda()
    # fixed_seg = torch.roll(fixed_seg, shifts=-n, dims=3)  # 4
    # fixed_seg[:,:,:,:-n, :] = 0
    # fixed_seg = flip(fixed_seg[0].cpu()).unsqueeze(0).cuda()
    

    # -- rescale imgs here -- #
    rescale = tio.RescaleIntensity(out_min_max=(0, 1), percentiles=(0.5, 99.5))
    input_moving = rescale(input_moving[0].cpu()).unsqueeze(0).cuda()
    input_fixed = rescale(input_fixed[0].cpu()).unsqueeze(0).cuda()
    
    # predict
    if args.seg:
        y_seg = model(input_fixed)
        y_seg = torch.nn.functional.softmax(y_seg, dim=1).float()
    else:
        if args.nca:
            moved, warp, y_seg = model(input_moving, input_fixed, input_seg, registration=True)#, inference=True)
        else:
            moved, warp, y_seg = model(input_moving, input_fixed, input_seg, registration=True)

        # Black-White grid and warp for later plotting
        # g = pystrum.pynd.ndutils.bw_grid(input_moving.size()[2:], 1)[np.newaxis, np.newaxis, ...]
        # g = torch.from_numpy(g).to(device).float()
        # g = model.transformer(g, warp)

        # save images
        moved = moved.permute(0, 2, 3, 4, 1).detach().cpu().numpy().squeeze()#.transpose(0, 2, 1)
        # y_seg = y_seg.permute(0, 2, 3, 4, 1).detach().cpu().numpy().squeeze()#.transpose(0, 2, 1)
        # g = g.detach().cpu().numpy().squeeze()#.transpose(0, 2, 1)
        warp = warp.permute(0, 2, 3, 4, 1).detach().cpu().numpy().squeeze()
        # y_seg[y_seg != 0] = 1
        

        y_seg = y_seg.argmax(1).detach().cpu().numpy().squeeze().transpose(2, 1, 0).astype(float)

        sitk.WriteImage(sitk.GetImageFromArray(moved.transpose(2, 0, 1).swapaxes(-2,-1)[...,::-1]), os.path.join(args.out, 'moved_img.nii.gz'))
        sitk.WriteImage(sitk.GetImageFromArray(warp.transpose(2, 0, 1, 3).swapaxes(-3,-2)[...,::-1, :]), os.path.join(args.out, 'flow.nii.gz'))
        # np.save(os.path.join(args.out, 'flow_grid'), g)
        sitk.WriteImage(sitk.GetImageFromArray(y_seg), os.path.join(args.out, 'moved_seg.nii.gz'))

        # -- Only when augmentations are done, otherwise just copy the moving img -- #
        # shutil.copy(args.moving, os.path.join(args.out, 'moving_img.nii.gz'))
        # shutil.copy(args.moving_seg, os.path.join(args.out, 'moving_seg.nii.gz'))

        input_moving = input_moving.detach().cpu().numpy().squeeze()
        sitk.WriteImage(sitk.GetImageFromArray(input_moving.transpose(2, 0, 1).swapaxes(-2,-1)[...,::-1]), os.path.join(args.out, 'moving_img.nii.gz'))
        input_seg = input_seg.detach().cpu().numpy().squeeze().transpose(2, 1, 0).astype(float)
        sitk.WriteImage(sitk.GetImageFromArray(input_seg), os.path.join(args.out, 'moving_seg.nii.gz'))
        fixed_seg = fixed_seg.detach().cpu().numpy().squeeze().transpose(2, 1, 0).astype(float)
        sitk.WriteImage(sitk.GetImageFromArray(fixed_seg), os.path.join(args.out, 'fixed_seg.nii.gz'))

    input_fixed = input_fixed.detach().cpu().numpy().squeeze()
    sitk.WriteImage(sitk.GetImageFromArray(input_fixed.transpose(2, 0, 1).swapaxes(-2,-1)[...,::-1]), os.path.join(args.out, 'fixed_img.nii.gz'))

    # shutil.copy(args.fixed, os.path.join(args.out, 'fixed_img.nii.gz' if not args.seg else 'img.nii.gz'))
    # if args.gt:
    #     shutil.copy(args.gt, os.path.join(args.out, 'fixed_seg.nii.gz' if not args.seg else 'seg_gt.nii.gz'))

    if args.gt and args.seg:
        shutil.copy(args.gt, os.path.join(args.out, 'seg_gt.nii.gz'))

    if args.seg:
        y_seg = y_seg.argmax(1).detach().cpu().numpy().squeeze().transpose(2, 1, 0).astype(float)
        sitk.WriteImage(sitk.GetImageFromArray(y_seg), os.path.join(args.out, 'pred_seg.nii.gz'))
    
# -- Main function for setup execution -- #
def main():
    register()

if __name__ == "__main__":
    register()