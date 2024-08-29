#!/usr/bin/env python

"""
Example script to train a VoxelMorph model.

You will likely have to customize this script slightly to accommodate your own data. All images
should be appropriately cropped and scaled to values between 0 and 1.

If an atlas file is provided with the --atlas flag, then scan-to-atlas training is performed.
Otherwise, registration will be scan-to-scan.

If you use this code, please cite the following, and read function docs for further info/citations.

    VoxelMorph: A Learning Framework for Deformable Medical Image Registration G. Balakrishnan, A.
    Zhao, M. R. Sabuncu, J. Guttag, A.V. Dalca. IEEE TMI: Transactions on Medical Imaging. 38(8). pp
    1788-1800. 2019. 

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
implied. See the License for the specific language governing permissions and limitations under the
License.
"""

# python train.py --img-list //local/scratch/aranem/BMVC_2024/BMVC_2024_predictions/VoxelMorph_rigid/train_list.txt --seg-list //local/scratch/aranem/BMVC_2024/BMVC_2024_predictions/VoxelMorph_rigid/train_seg.txt


"""
Code has been adapted so segmentations are warped during training as well. Build the list files manually and split them as well so we can make proper
validation at the end. Don't forget to set model-dir correctly.
Remember that the input images has to be affinely aligned. The moving seg is anligned using te image transformation flow, i.e. without considering the GT.
Normalization between [0, 1] takes place during training!
"""

import os, monai
import argparse
import time, random
import numpy as np
import torch
from monai.metrics import DiceMetric
from scripts.validate import validate
from tqdm import tqdm 
from voxelmorph.torch.utils import *
import logging

# import voxelmorph with pytorch backend
os.environ['VXM_BACKEND'] = 'pytorch'
os.environ['NEURITE_BACKEND'] = 'pytorch'
import voxelmorph as vxm  # nopep8

# Set seeds for numpy, random and pytorch
set_all_seeds(3299)
# torch.set_printoptions(profile="full")

# -- Set patch size and nr of patches to be extracted -- #
PATCH_SIZE = 32
NR_PATCHES = 5

def train():
    args = get_args()
    _train(args)

def get_args():
    # parse the commandline
    parser = argparse.ArgumentParser()

    # data organization parameters
    parser.add_argument('--img-list', required=True, help='line-seperated list of training files')
    parser.add_argument('--img-prefix', help='optional input image file prefix')
    parser.add_argument('--img-suffix', help='optional input image file suffix')
    parser.add_argument('--seg-list', required=True, help='line-seperated list of training segs')
    parser.add_argument('--seg-prefix', help='optional input seg file prefix')
    parser.add_argument('--seg-suffix', help='optional input seg file suffix')
    parser.add_argument('--atlas', help='atlas filename (default: data/atlas_norm.nii)')
    # parser.add_argument('--model-dir', default='//local/scratch/aranem/BMVC_2024/BMVC_2024_predictions/VoxelMorph_rigid/vxm_torch_250_113_ncc_ce',
    parser.add_argument('--model-dir', default='//local/scratch/aranem/BMVC_2024/BMVC_2024_predictions/VoxelMorph_rigid/vxm_torch_250_110_111_112_113_ncc_ce',
    # parser.add_argument('--model-dir', default='//local/scratch/aranem/BMVC_2024/BMVC_2024_predictions/UNet_VxM/unet_torch_250_110_ce',
                        help='model output directory.')
    parser.add_argument('--multichannel', action='store_true',
                        help='specify that data has multiple channels')
    parser.add_argument('--seg', action='store_true',
                        help='Set this to use VxM U-Net as segmentatin network.')
    parser.add_argument('--nca', action='store_true',
                        help='Set this to use VxM NCA for registration.')
    parser.add_argument('--patches', action='store_true',
                        help='Set this to use VxM NCA for registration with patches during training only.')
    parser.add_argument('--transmorph', action='store_true',
                        help='Set this to use TransMorph for registration.')
    parser.add_argument('--vitvnet', action='store_true',
                        help='Set this to use ViTVNet for registration.')
    parser.add_argument('--nicetrans', action='store_true',
                        help='Set this to use NICE-Trans for registration.')

    # training parameters
    parser.add_argument('--gpu', default='0', help='GPU ID number(s), comma-separated (default: 0)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size (default: 1)')
    parser.add_argument('--epochs', type=int, default=250,
                        help='number of training epochs (default: 250)')
    parser.add_argument('--steps-per-epoch', type=int, default=250,
                        help='frequency of model saves (default: 100)')
    parser.add_argument('--load-model', help='optional model file to initialize with')
    parser.add_argument('--initial-epoch', type=int, default=0,
                        help='initial epoch number (default: 0)')
    # parser.add_argument('--lr', type=float, default=16e-4, help='learning rate (default: 1e-4)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
    parser.add_argument('--cudnn-nondet', action='store_true',
                        help='disable cudnn determinism - might slow down training')

    # network architecture parameters
    parser.add_argument('--enc', type=int, nargs='+',
                        help='list of unet encoder filters (default: 16 32 32 32)')
    parser.add_argument('--dec', type=int, nargs='+',
                        help='list of unet decorder filters (default: 32 32 32 32 32 16 16)')
    parser.add_argument('--int-steps', type=int, default=7,
                        help='number of integration steps (default: 7)')
    parser.add_argument('--int-downsize', type=int, default=2,
                        help='flow downsample factor for integration (default: 2)')
    parser.add_argument('--bidir', action='store_true', help='enable bidirectional cost function')

    # loss hyperparameters
    parser.add_argument('--image-loss', default='ncc',
                        help='image reconstruction loss - can be mse or ncc (default: mse)')
    parser.add_argument('--lambda', type=float, dest='weight', default=0.01,
                        help='weight of deformation loss (default: 0.01)')
    
    # NCA related params
    parser.add_argument('--kernel_size', type=int, default=7,
                        help='Kernel sie for NCA (default: 7)')
    parser.add_argument('--steps', type=int, default=10,
                        help='Number of steps for NCA (default: 10)')
    parser.add_argument('--fire_rate', type=float, default=0.5,
                        help='Fire ratefor NCA (default: 0.5)')
    parser.add_argument('--n_channels', type=int, default=16,
                        help='Number of channels for NCA (default: 16)')
    parser.add_argument('--hidden_size', type=int, default=64,
                        help='Number of hidden size (default: 64)')
    args = parser.parse_args()
    return args

def _train(args):
    bidir = False #args.bidir

    # load and prepare training data
    train_files = vxm.py.utils.read_file_list(args.img_list, prefix=args.img_prefix,
                                              suffix=args.img_suffix)
    seg_files = vxm.py.utils.read_file_list(args.seg_list, prefix=args.seg_prefix,
                                            suffix=args.seg_suffix)
    assert len(train_files) > 0, 'Could not find any training data.'
    assert len(seg_files) > 0, 'Could not find any segmentations.'
    assert not(args.seg and args.nca), "Please only set one, either seg for segmentation with UNet or nca for registration but not both!"

    # Shuffle the img and seg lists equally
    c = list(zip(train_files, seg_files))
    random.shuffle(c)
    train_files, seg_files = zip(*c)

    # -- Split into train and val based on tasks using 80:20 split -- #
    train_files_train = train_files[:int((len(train_files)+1)*.80)]  # Remaining 80% to training set
    seg_files_train = seg_files[:int((len(seg_files)+1)*.80)]  # Remaining 80% to training set
    train_files_val = train_files[int((len(train_files)+1)*.80):] # Splits 20% data to test set
    seg_files_val = seg_files[int((len(seg_files)+1)*.80):] # Splits 20% data to test set

    # no need to append an extra feature axis if data is multichannel
    add_feat_axis = True #not args.multichannel

    if args.atlas:
        # scan-to-atlas generator
        atlas = vxm.py.utils.load_volfile(args.atlas, np_var='vol',
                                          add_batch_axis=True, add_feat_axis=add_feat_axis)
        generator = vxm.generators.scan_to_atlas(train_files_train, atlas,
                                                 batch_size=args.batch_size, bidir=args.bidir,
                                                 add_feat_axis=add_feat_axis)
    else:
        # scan-to-scan generator
        generator = vxm.generators.scan_to_scan(
            train_files_train, seg_files_train, batch_size=args.batch_size, bidir=args.bidir, add_feat_axis=add_feat_axis)

    # extract shape from sampled input
    inshape = next(generator)[0][0].shape[1:-1]

    # prepare model folder
    model_dir = args.model_dir
    os.makedirs(model_dir, exist_ok=True)

    # device handling
    gpus = args.gpu.split(',')
    nb_gpus = len(gpus)
    device = 'cuda'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    assert np.mod(args.batch_size, nb_gpus) == 0, \
        'Batch size (%d) should be a multiple of the nr of gpus (%d)' % (args.batch_size, nb_devices)

    # enabling cudnn determinism appears to speed up training by a lot
    torch.backends.cudnn.deterministic = not args.cudnn_nondet

    # unet architecture
    enc_nf = args.enc if args.enc else [16, 32, 32, 32]
    dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]
    
    if args.load_model:
        # load initial model (if specified)
        if args.nicetrans:
            model = vxm.networks.NICE_Trans.load(args.load_model, device)
        elif args.vitvnet:
            model = vxm.networks.ViTVNet.load(args.load_model, device)
        elif args.transmorph:
            model = vxm.networks.TransMorph.load(args.load_model, device)
        elif args.nca:
            model = vxm.networks.NCAMorph.load(args.load_model, device)
        elif args.seg:
            model = vxm.networks.UNet.load(args.load_model, device)
        else:
            model = vxm.networks.VxmDense.load(args.load_model, device)
    else:
        if args.nicetrans:
            model = vxm.networks.NICE_Trans()
        elif args.vitvnet:
            model = vxm.networks.ViTVNet(img_size=inshape, int_steps=args.int_steps)
        elif args.transmorph:
            model = vxm.networks.TransMorph(img_size=inshape)
        elif args.nca:
            if args.patches:
                model = vxm.networks.NCAMorph(inshape=(32,32,32), kernel_size = args.kernel_size, steps = args.steps, fire_rate = args.fire_rate, n_channels = args.n_channels, hidden_size = args.hidden_size, full_shape=inshape)
            else:
                model = vxm.networks.NCAMorph(inshape=inshape, kernel_size = args.kernel_size, steps = args.steps, fire_rate = args.fire_rate, n_channels = args.n_channels, hidden_size = args.hidden_size, full_shape=inshape)
        elif args.seg:
            model = vxm.networks.UNet(
                inshape=inshape,
                infeats=1,  # <-- One input feature
                nb_features=[enc_nf, dec_nf],
                nb_levels=None,
                feat_mult=1,
                feat_out=2, # <-- 2 output features (binary)
                nb_conv_per_level=1,
                half_res=False,
            )
        else:
            # otherwise configure new model
            model = vxm.networks.VxmDense(
                inshape=inshape,
                nb_unet_features=[enc_nf, dec_nf],
                bidir=bidir,
                int_steps=args.int_steps,
                int_downsize=args.int_downsize
            )

    if nb_gpus > 1:
        # use multiple GPUs via DataParallel
        model = torch.nn.DataParallel(model)
        model.save = model.module.save

    # prepare the model for training and send to device
    model.to(device)
    model.train()

    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # set logger to also log the messages
    logging.basicConfig(filename = os.path.join(model_dir,"train_log.txt"), 
                        format = '%(asctime)s %(message)s') 

    logger = logging.getLogger() 
    logger.setLevel(logging.DEBUG)

    # prepare image loss        
    if args.seg:
        losses = [monai.losses.DiceCELoss(sigmoid=True, reduction='mean', to_onehot_y=True)]
        weights = [1]
    else:
        if args.image_loss == 'ncc':
            image_loss_func = vxm.losses.NCC().loss
        elif args.image_loss == 'mse':
            image_loss_func = vxm.losses.MSE().loss
        else:
            raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % args.image_loss)

        # need two image loss functions if bidirectional
        if bidir:
            losses = [image_loss_func, image_loss_func]
            weights = [0.5, 0.5]
        else:
            losses = [image_loss_func]
            weights = [1]

        # prepare deformation loss
        losses += [vxm.losses.Grad('l2', loss_mult=args.int_downsize).loss]
        weights += [args.weight]    # 0.01

        # Dice-CE Loss
        losses += [monai.losses.DiceCELoss(sigmoid=True, reduction='mean', to_onehot_y=True)]
        weights += [1]

    # training loops
    for epoch in range(args.initial_epoch, args.epochs):

        # save model checkpoint
        if epoch % 150 == 0:
            model.save(os.path.join(model_dir, '%04d.pt' % epoch))
            validate(model, train_files_val, seg_files_val, epoch, out_=os.path.join(args.model_dir, "validation"), seg=args.seg)

        epoch_loss = []
        epoch_total_loss = []
        epoch_step_time = []
        dice = []
        dice_nm = []

        # for step in range(steps):
        for _ in range(args.steps_per_epoch):
        # for step in tqdm(range(args.steps_per_epoch)):

            step_start_time = time.time()

            # generate inputs (and true outputs) and convert them to tensors
            inputs, y_true, segs = next(generator)
            inputs = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in inputs]
            y_true = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in y_true]    # fixed image
            segs = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in segs]

            # -- Select n patches if patches for NCA -- #
            if args.nca and args.patches:
                # -- get random patch indices of PATCH_SIZExPATCH_SIZExPATCH_SIZE patches -- #
                idxs, inputs_, y_true_, segs_ = list(), list(), list(), list()
                for _ in range(NR_PATCHES):   # --> extract NR_PATCHES patches randomly
                    idxs.append([random.randint(0, inshape[0] -  PATCH_SIZE), random.randint(0, inshape[1] -  PATCH_SIZE), random.randint(0, inshape[2] -  PATCH_SIZE)])
                for patch in idxs:
                    inputs_.append([x[:, :, patch[0]:patch[0]+PATCH_SIZE, patch[1]:patch[1]+PATCH_SIZE, patch[2]:patch[2]+PATCH_SIZE] for x in inputs]) #[torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in inputs]
                    y_true_.append([x[:, :, patch[0]:patch[0]+PATCH_SIZE, patch[1]:patch[1]+PATCH_SIZE, patch[2]:patch[2]+PATCH_SIZE] for x in y_true]) #[torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in y_true]
                    segs_.append([x[:, :, patch[0]:patch[0]+PATCH_SIZE, patch[1]:patch[1]+PATCH_SIZE, patch[2]:patch[2]+PATCH_SIZE] for x in segs]) #[torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in segs]
                # -- Join the patches along batch dimension (0) -- #
                inputs = [torch.stack(x, dim=0).squeeze(1) for x in  zip(*inputs_)]
                y_true = [torch.stack(x, dim=0).squeeze(1) for x in  zip(*y_true_)]
                segs = [torch.stack(x, dim=0).squeeze(1) for x in  zip(*segs_)]
                
                # - Uncomment this if only one patch is extracted during training -- #
                # idxs = [random.randint(0, inshape[0] -  PATCH_SIZE), random.randint(0, inshape[1] -  PATCH_SIZE), random.randint(0, inshape[2] -  PATCH_SIZE)]
                # inputs = [x[:, :, idxs[0]:idxs[0]+PATCH_SIZE, idxs[1]:idxs[1]+PATCH_SIZE, idxs[2]:idxs[2]+PATCH_SIZE] for x in inputs] #[torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in inputs]
                # y_true = [x[:, :, idxs[0]:idxs[0]+PATCH_SIZE, idxs[1]:idxs[1]+PATCH_SIZE, idxs[2]:idxs[2]+PATCH_SIZE] for x in y_true] #[torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in y_true]
                # segs = [x[:, :, idxs[0]:idxs[0]+PATCH_SIZE, idxs[1]:idxs[1]+PATCH_SIZE, idxs[2]:idxs[2]+PATCH_SIZE] for x in segs] #[torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in segs]

            loss = 0
            loss_list = []

            # run inputs through the model to produce a warped image and flow field
            if args.seg:
                # TODO: Does not work, y to one hot fails somehow... --> monai
                y_pred_ = model(inputs[1])   # <-- Only target into U-net, segmentation out
                # Switch first entries so CE loss gets pred first and then the target, otherwise it gets the wrong arguments!
                y_true = [segs[1]]#, torch.nn.functional.softmax(y_pred_, dim=1).float()]
                y_pred = [y_pred_]
            else:
                # -- Prepare for losses (NCC, L2, DiceCE) -- #
                y_pred = model(*inputs, segs[0]) # <-- modified, return moved_im, flow, moved_seg
                y_true += [None, segs[1]] # second element for CEL, gt segmentation, skip third one, set to None
                y_pred_ = y_pred[-1].clone()    # Clone the y_seg mask
                y_pred = [*y_pred[:-1], y_pred_]   # moved_im, flow, y_seg

            # calculate total loss
            for n, loss_function in enumerate(losses):
                curr_loss = loss_function(y_pred[n], y_true[n]) * weights[n]
                loss_list.append(curr_loss.item())
                loss += curr_loss

            epoch_loss.append(loss_list)
            epoch_total_loss.append(loss.item())

            # backpropagate and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate Dice score
            dice_ = DiceMetric(include_background=False, ignore_empty=False)(torch.argmax(y_pred_, axis=1).unsqueeze(0), segs[1])
            dice.append(np.mean(dice_.cpu().numpy())*100)

            if not args.seg:
                mean_dice_nm = DiceMetric(include_background=False, ignore_empty=False)(segs[0], segs[1])
                dice_nm.append(np.mean(mean_dice_nm.cpu().numpy())*100)

            # get compute time
            epoch_step_time.append(time.time() - step_start_time)

        # print epoch info
        epoch_info = 'Epoch %d/%d' % (epoch + 1, args.epochs)
        time_step_info = '%.4f sec/step' % np.mean(epoch_step_time)
        time_epoch_info = '%.4f sec/epoch' % np.sum(epoch_step_time)
        losses_info = ', '.join(['%.4e' % f for f in np.mean(epoch_loss, axis=0)])
        loss_info = 'loss: %.4e  (%s)' % (np.mean(epoch_total_loss), losses_info)
        dice_info = 'dice: %.4f' % (np.mean(dice))
        if not args.seg:
            dice_info_nm = 'dice (nm): %.4f' % (np.mean(dice_nm))
            print(' - '.join((epoch_info, time_step_info, time_epoch_info, loss_info, dice_info, dice_info_nm)), flush=True)
            logger.debug(' - '.join((epoch_info, time_step_info, time_epoch_info, loss_info, dice_info, dice_info_nm)))
        else:
            print(' - '.join((epoch_info, time_step_info, time_epoch_info, loss_info, dice_info)), flush=True)
            logger.debug(' - '.join((epoch_info, time_step_info, time_epoch_info, loss_info, dice_info)))

    tot_params, train_params = get_nr_parameters(model)
    model_size = get_model_size(model)
    logger.debug("Nr of parameter (total -- trainable): {} -- {}".format(tot_params, train_params))
    logger.debug("Model size in MB: {:.4f}".format(model_size))

    # final model save
    model.save(os.path.join(model_dir, '%04d.pt' % args.epochs))
    validate(model, train_files_val, seg_files_val, args.epochs, out_=os.path.join(args.model_dir, "validation"), seg=args.seg)

    
# -- Main function for setup execution -- #
def main():
    train()

if __name__ == "__main__":
    train()