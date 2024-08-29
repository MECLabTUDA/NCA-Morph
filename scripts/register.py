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


# python register.py --moving /home/aranem_locale/Desktop/mnts/local/scratch/aranem/Lifelong-nnUNet-storage/MICCAI_2023/MICCAI_2023_raw_data/nnUNet_ready_with_ts_folder/Task110_RUNMC/imagesTr/Case08_0001.nii.gz --fixed /home/aranem_locale/Desktop/mnts/local/scratch/aranem/Lifelong-nnUNet-storage/MICCAI_2023/MICCAI_2023_raw_data/nnUNet_ready_with_ts_folder/Task110_RUNMC/imagesTr/Case08_0000.nii.gz --moving_seg /home/aranem_locale/Desktop/mnts/local/scratch/aranem/Lifelong-nnUNet-storage/MICCAI_2023/MICCAI_2023_raw_data/nnUNet_ready_with_ts_folder_BACKUP/Task110_RUNMC/labelsTr/Case08_0001.nii.gz --model /home/aranem_locale/Desktop/MICCAI_2023/experiments/vxm_torch_250_110_ncc_dice/0200.pt --out /home/aranem_locale/Desktop/MICCAI_2023/experiments/vxm_torch_250_110_ncc_dice/eval/Case08/ --gt /home/aranem_locale/Desktop/mnts/local/scratch/aranem/Lifelong-nnUNet-storage/MICCAI_2023/MICCAI_2023_raw_data/nnUNet_ready_with_ts_folder_BACKUP/Task110_RUNMC/labelsTr/Case08_0000.nii.gz


import os
import argparse
from scripts.validate import validate
from voxelmorph.torch.utils import set_all_seeds

# import voxelmorph with pytorch backend
os.environ['VXM_BACKEND'] = 'pytorch'
os.environ['NEURITE_BACKEND'] = 'pytorch'
import voxelmorph as vxm   # nopep8

# Set seeds for numpy, random and pytorch
set_all_seeds(3299)

def register():
    args = get_args()
    return _register(args)

def get_args():
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
    return args

def _register(args):
    assert not(args.seg and args.nca), "Please only set one, either seg for segmentation with UNet or nca for registration but not both!"

    # device handling
    device = 'cuda'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # load and set up model
    if args.nicetrans:
        model = vxm.networks.NICE_Trans.load(args.model, device)
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

    # Extract meta information on model
    epoch = int(args.model.split(os.sep)[-1][:-3])
    model_dir = os.path.join(os.sep, *args.model.split(os.sep)[:-1])

    # Do validation for one sample based on args.npz
    val_res = validate(model, [args.fixed], [args.gt], epoch, out_=os.path.join(model_dir, "predictions"), seg=False, include_epoch=False)
    
    return val_res.loc[:, 'Dice'].mean(), val_res.loc[:, 'IoU'].mean(), val_res.loc[:, 'Dice nm'].mean(), val_res.loc[:, 'IoU nm'].mean(), val_res.loc[:, 'MSE (moving_to_fixed)'].mean()
    
# -- Main function for setup execution -- #
def main():
    register()

if __name__ == "__main__":
    register()