import os
import pandas as pd
import numpy as np
import torch
from monai.metrics import DiceMetric, MeanIoU, MSEMetric
from voxelmorph.torch.utils import *
import torchio as tio
import voxelmorph as vxm  # nopep8

def validate(model, train_files_val, seg_files_val, epoch, out_, seg=False, include_epoch=True):
    r"""
    Call this function in order to validate the model with data from the provided generator.
        :param model: The model architecture to use for predictions.
        :param train_files_val: List of paths to validation cases.
        :param seg_files_val: List of paths to seg validation cases.
        :param epoch: Integer of the current epoch we perform the validation in
        :param out_: Path to where the metrics (and samples) should be stored
        :param seg: If simple U-Net/Segmentation network is used (input of model will be different then..)
    """
    # Put model in eval mode and initialize dictionaries
    model.eval()
    val_res = pd.DataFrame()
    os.makedirs(out_, exist_ok=True)
    device = next(model.parameters()).device

    load_params = dict(np_var='vol', add_batch_axis=True, pad_shape=None, resize_factor=1, add_feat_axis=True,
                       rescale=tio.RescaleIntensity(out_min_max=(0, 1), percentiles=(0.5, 99.5)))
    load_params_seg = dict(np_var='vol', add_batch_axis=True, pad_shape=None, resize_factor=1, add_feat_axis=True,
                       rescale=None)
    # with torch.no_grad():
    for idx, nii_f in enumerate(train_files_val):

        nii_m = nii_f.replace('_0000.nii', '_0001.nii')
        seg_m = seg_files_val[idx].replace('_0000.nii', '_0001.nii')

        # Extract the case name
        case = nii_f.split(os.sep)[-1].split('.nii.gz')[0][:-5] # Remove _0000
        task = nii_f.split(os.sep)[-3]
        if include_epoch:
            out = os.path.join(out_, "Epoch_"+str(epoch), task, case)
        else:
            out = os.path.join(out_, task, case)

        os.makedirs(out, exist_ok=True)

        # Load input image (in this case we already pre-processed it, i.e. extracted embeddings and fixed shape issues using one of our scripts)
        imgs_ = vxm.py.utils.load_volfile(nii_f, **load_params)
        segs_ = vxm.py.utils.load_volfile(seg_files_val[idx], **load_params_seg)
        imgs_m = vxm.py.utils.load_volfile(nii_m, **load_params)
        segs_m = vxm.py.utils.load_volfile(seg_m, **load_params_seg)
        
        # To torch
        imgs = torch.from_numpy(imgs_).to(device).float().permute(0, 4, 1, 2, 3)
        segs = torch.from_numpy(segs_).to(device).float().permute(0, 4, 1, 2, 3)
        imgs_m = torch.from_numpy(imgs_m).to(device).float().permute(0, 4, 1, 2, 3)
        segs_m = torch.from_numpy(segs_m).to(device).float().permute(0, 4, 1, 2, 3)

        # Predict
        if seg:
            y_seg = model(imgs)   # <-- Only target into U-net, segmentation out
        else:
            y_source, pos_flow, y_seg = model(imgs_m, imgs, segs_m)

        # Calculate metrics
        Dice = DiceMetric(include_background=False, ignore_empty=False)(y_seg.argmax(1).unsqueeze(0), segs)
        IoU = MeanIoU(include_background=False, ignore_empty=False)(y_seg.argmax(1).unsqueeze(0), segs)

        if not seg:
            MSE_imgs = MSEMetric()(y_source, imgs)
            IoU_ = MeanIoU(include_background=False, ignore_empty=False)(segs_m, segs)
            Dice_ = DiceMetric(include_background=False, ignore_empty=False)(segs_m, segs)

        # Append to dataframe
        if seg:
            val_res = pd.concat([val_res,
                                  pd.DataFrame.from_records([{'Epoch': str(epoch), 'Task': task, 'ID': case,
                                                              'Dice': np.mean(Dice.cpu().numpy()*100),
                                                              'IoU': np.mean(IoU.cpu().numpy()*100),
                                                              }])
                            ], axis=0)
        else:
            val_res = pd.concat([val_res,
                                  pd.DataFrame.from_records([{'Epoch': str(epoch), 'Task': task, 'ID': case,
                                                              'Dice': np.mean(Dice.cpu().numpy()*100),
                                                              'Dice nm': np.mean(Dice_.cpu().numpy()*100),
                                                              'IoU': np.mean(IoU.cpu().numpy()*100),
                                                              'IoU nm': np.mean(IoU_.cpu().numpy()*100),
                                                              'MSE (moving_to_fixed)': np.mean(MSE_imgs.cpu().numpy())
                                                              }])
                            ], axis=0)
        
        # Store predictions with flows
            y_source = y_source.permute(0, 2, 3, 4, 1).detach().cpu().numpy().squeeze()
            pos_flow = pos_flow.permute(0, 2, 3, 4, 1).detach().cpu().numpy().squeeze()
        y_seg = y_seg.argmax(1).detach().cpu().numpy().squeeze().transpose(2, 1, 0).astype(float)

        segs = segs.detach().cpu().numpy().squeeze().transpose(2, 1, 0).astype(float)   # fixed_seg
        imgs = imgs.detach().cpu().numpy().squeeze()
        segs_m = segs_m.detach().cpu().numpy().squeeze().transpose(2, 1, 0).astype(float) # moving_seg
        imgs_m = imgs_m.detach().cpu().numpy().squeeze()
              
        if seg:
            sitk.WriteImage(sitk.GetImageFromArray(y_seg), os.path.join(out, 'pred.nii.gz'))  # prediction
            sitk.WriteImage(sitk.GetImageFromArray(segs), os.path.join(out, 'gt.nii.gz'))   # GT
            sitk.WriteImage(sitk.GetImageFromArray(imgs.transpose(2, 0, 1).swapaxes(-2,-1)[...,::-1]), os.path.join(out, 'img.nii.gz')) # image
        else:
            sitk.WriteImage(sitk.GetImageFromArray(y_source.transpose(2, 0, 1).swapaxes(-2,-1)[...,::-1]), os.path.join(out, 'moved_img.nii.gz'))  # moved img
            sitk.WriteImage(sitk.GetImageFromArray(pos_flow.transpose(2, 0, 1, 3).swapaxes(-3,-2)[...,::-1, :]), os.path.join(out, 'flow.nii.gz')) # flow
            sitk.WriteImage(sitk.GetImageFromArray(y_seg), os.path.join(out, 'moved_seg.nii.gz'))  # prediction
            sitk.WriteImage(sitk.GetImageFromArray(imgs_m.transpose(2, 0, 1).swapaxes(-2,-1)[...,::-1]), os.path.join(out, 'moving_img.nii.gz'))   # moving img
            sitk.WriteImage(sitk.GetImageFromArray(segs_m), os.path.join(out, 'moving_seg.nii.gz'))
            sitk.WriteImage(sitk.GetImageFromArray(segs), os.path.join(out, 'fixed_seg.nii.gz'))
            sitk.WriteImage(sitk.GetImageFromArray(imgs.transpose(2, 0, 1).swapaxes(-2,-1)[...,::-1]), os.path.join(out, 'fixed_img.nii.gz')) # image
        
    # Store metrics csv
    if not os.path.isfile(os.path.join(out_, "validation_results.csv")):
        val_res.to_csv(os.path.join(out_, "validation_results.csv"), index=False, sep=',')  # <-- Includes the header
    else: # else it exists so append without writing the header
        val_res.to_csv(os.path.join(out_, "validation_results.csv"), index=False, sep=',', mode='a', header=False)  # <-- Omits the header 

    # Put model back into train mode and return the results
    model.train()
    return val_res