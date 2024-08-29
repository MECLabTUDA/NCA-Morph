# import ants
import SimpleITK as sitk
import numpy as np
import pandas as pd
import os, ants, torch
from tqdm import tqdm
import torchio as tio
from monai.metrics import DiceMetric, MeanIoU, MSEMetric

# https://antspy.readthedocs.io/en/latest/registration.html

def rigid_align(fixed, moving, seg_m):
    r"""All inputs should be sitk Image objects not numpy arrays.
    """
    initial_transform = sitk.CenteredTransformInitializer(fixed, moving,
                                                          sitk.Euler3DTransform(),
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY
                                                         )
    moving = sitk.Resample(moving, fixed, initial_transform, sitk.sitkLinear, 0.0, moving.GetPixelID())
    # -- Use the transform between the images to apply it onto the segmentation as we don't have GT during inference -- #
    seg_m = sitk.Resample(seg_m, fixed, initial_transform, sitk.sitkLinear, 0.0, seg_m.GetPixelID())
    
    return sitk.GetArrayFromImage(moving), sitk.GetArrayFromImage(seg_m)

def affine_align(fixed, moving, seg_m):
    r"""All inputs should be sitk Image objects not numpy arrays.
    """
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(fixed)
    elastixImageFilter.SetMovingImage(moving)
    elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap("affine"))
    elastixImageFilter.Execute()
    moving = sitk.WriteImage(elastixImageFilter.GetResultImage())

    elastixImageFilter.SetMovingImage(seg_m)
    elastixImageFilter.Execute()
    seg_m = sitk.WriteImage(elastixImageFilter.GetResultImage())
    
    return sitk.GetArrayFromImage(moving), sitk.GetArrayFromImage(seg_m)

def bspline_align(fixed, moving, seg_m):
    r"""All inputs should be sitk Image objects not numpy arrays.
    """
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(fixed)
    elastixImageFilter.SetMovingImage(moving)

    parameterMapVector = sitk.VectorOfParameterMap()
    parameterMap = sitk.GetDefaultParameterMap("affine")
    # parameterMap['MaximumNumberOfIterations'] = ['512']
    # parameterMap['DefaultPixelValue'] = ['-1000']  # AIR in Housenfield units
    parameterMapVector.append(parameterMap)
    parameterMap = sitk.GetDefaultParameterMap("bspline")
    # parameterMap['MaximumNumberOfIterations'] = ['512']
    # parameterMap['DefaultPixelValue'] = ['-1000']  # AIR in Housenfield units
    parameterMapVector.append(parameterMap)
    # parameterMapVector.append(sitk.GetDefaultParameterMap("affine"))
    # parameterMapVector.append(sitk.GetDefaultParameterMap("bspline"))
    elastixImageFilter.SetParameterMap(parameterMapVector)

    elastixImageFilter.Execute()
    moving = sitk.WriteImage(elastixImageFilter.GetResultImage())

    elastixImageFilter.SetMovingImage(seg_m)
    elastixImageFilter.Execute()
    seg_m = sitk.WriteImage(elastixImageFilter.GetResultImage())
    
    return sitk.GetArrayFromImage(moving), sitk.GetArrayFromImage(seg_m)

# def affine_align(fixed, moving, seg_m):
#     r"""All inputs should be ant Image objects not numpy arrays.
#     """
#     mytx = ants.registration(fixed=fixed, moving=moving, type_of_transform='Affine', verbose=True)
#     # mytx = ants.registration(fixed=fixed, moving=moving, type_of_transform='AffineFast', verbose=True)
#     moving = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=mytx['fwdtransforms'])
#     seg_m = ants.apply_transforms(fixed=fixed, moving=seg_m, transformlist=mytx['fwdtransforms'])
#     return moving.numpy(), seg_m.numpy()

def syn_align(fixed, moving, seg_m):
    r"""All inputs should be ant Image objects not numpy arrays.
    """
    mytx = ants.registration(fixed=fixed, moving=moving, type_of_transform='SyN', verbose=True)
    moving = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=mytx['fwdtransforms'])
    seg_m = ants.apply_transforms(fixed=fixed, moving=seg_m, transformlist=mytx['fwdtransforms'])
    return moving.numpy(), seg_m.numpy()

def syncc_align(fixed, moving, seg_m):
    r"""All inputs should be ant Image objects not numpy arrays.
    """
    mytx = ants.registration(fixed=fixed, moving=moving, type_of_transform='SyNCC', verbose=True)
    moving = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=mytx['fwdtransforms'])
    seg_m = ants.apply_transforms(fixed=fixed, moving=seg_m, transformlist=mytx['fwdtransforms'])
    return moving.numpy(), seg_m.numpy()

def elasticsyn_align(fixed, moving, seg_m):
    r"""All inputs should be ant Image objects not numpy arrays.
    """
    mytx = ants.registration(fixed=fixed, moving=moving, type_of_transform='ElasticSyN', verbose=True)
    moving = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=mytx['fwdtransforms'])
    seg_m = ants.apply_transforms(fixed=fixed, moving=seg_m, transformlist=mytx['fwdtransforms'])
    return moving.numpy(), seg_m.numpy()

if __name__ == "__main__":
    paths = [
             '/media/aranem_locale/AR_subs_exps/MIDL_2024_NCA-Morph/MIDL_2024_raw_data/mapped_rigid/nnUNet_ready_with_ts_folder_rigid_aligned/Task210_OASIS',
             '/media/aranem_locale/AR_subs_exps/MIDL_2024_NCA-Morph/MIDL_2024_raw_data/mapped_rigid/nnUNet_ready_with_ts_folder_rigid_aligned/Task211_Prostate',
             '/media/aranem_locale/AR_subs_exps/MIDL_2024_NCA-Morph/MIDL_2024_raw_data/mapped_rigid/nnUNet_ready_with_ts_folder_rigid_aligned/Task212_Hippocampus',
            #  '/local/scratch/aranem/NeurIPS_2023/NeurIPS_2023_raw_data/mapped_rigid/nnUNet_ready_with_ts_folder_rigid_aligned/Task210_OASIS',
            #  '/local/scratch/aranem/NeurIPS_2023/NeurIPS_2023_raw_data/mapped_rigid/nnUNet_ready_with_ts_folder_rigid_aligned/Task211_Prostate',
            #  '/local/scratch/aranem/NeurIPS_2023/NeurIPS_2023_raw_data/mapped_rigid/nnUNet_ready_with_ts_folder_rigid_aligned/Task212_Hippocampus'
            #  '/local/scratch/aranem/NeurIPS_2023/NeurIPS_2023_raw_data/mapped_rigid/nnUNet_ready_with_ts_folder_rigid_aligned/Task213_Cardiac'
            ]

    outs =  [    
            #  '/media/aranem_locale/AR_subs_exps/MIDL_2024_NCA-Morph/MIDL_2024_trained_models/Rigid',
            #  '/media/aranem_locale/AR_subs_exps/MIDL_2024_NCA-Morph/MIDL_2024_trained_models/SyN',
             '/media/aranem_locale/AR_subs_exps/MIDL_2024_NCA-Morph/MIDL_2024_trained_models/SyNCC',
            #  '/media/aranem_locale/AR_subs_exps/MIDL_2024_NCA-Morph/MIDL_2024_trained_models/ElasticSyN',
            ]

    val_res = pd.DataFrame()

    # -- Ants weirdly flips the dims of the array, so reload and fix it with sitk since we know how to do it with sitk ;) -- #
    for idx, path in enumerate(paths):
        task_ = path.split('/')[-1]
        task = path.split('/')[-1].split('_')[0].replace('Task', '')
        out_ = os.path.join(outs[0], task, 'predictions', task_)
        os.makedirs(out_, exist_ok=True)
        cases = [x.split('_0000.nii.gz')[0] for x in os.listdir(os.path.join(path, 'imagesTs')) if '._' not in x and '.json' not in x and 'DS_Store' not in x and '_0001.nii.gz' not in x]
        cases = np.unique(cases)
        # -- Build fixed image, moving image and seg image pairs -- #
        for case in tqdm(cases):
            out_new = os.path.join(out_, case)
            os.makedirs(out_new, exist_ok=True)
            f, m, s, f_s = sitk.ReadImage(os.path.join(path, 'imagesTs', case+'_0000.nii.gz')), sitk.ReadImage(os.path.join(path, 'imagesTs', case+'_0001.nii.gz')), sitk.ReadImage(os.path.join(path, 'labelsTs', case+'_0001.nii.gz')), sitk.ReadImage(os.path.join(path, 'labelsTs', case+'_0000.nii.gz'))
            # s[s != 0] = 1 # <-- Keep it binary..
            # moving, seg = rigid_align(f, m, s)
            # moving, seg = affine_align(f, m, s)
            # moving, seg = bspline_align(f, m, s)
            # moving, seg = syn_align(ants.from_numpy(sitk.GetArrayFromImage(f)), ants.from_numpy(sitk.GetArrayFromImage(m)), ants.from_numpy(sitk.GetArrayFromImage(s)))
            moving, seg = syncc_align(ants.from_numpy(sitk.GetArrayFromImage(f)), ants.from_numpy(sitk.GetArrayFromImage(m)), ants.from_numpy(sitk.GetArrayFromImage(s)))
            # moving, seg = elasticsyn_align(ants.from_numpy(sitk.GetArrayFromImage(f)), ants.from_numpy(sitk.GetArrayFromImage(m)), ants.from_numpy(sitk.GetArrayFromImage(s)))
            f = sitk.GetArrayFromImage(f)
            s = sitk.GetArrayFromImage(s)
            f_s = sitk.GetArrayFromImage(f_s)
            
            # -- Calculate metrics -- #
            MSE_imgs = MSEMetric()(torch.Tensor(f), torch.Tensor(moving))
            Dice = DiceMetric(include_background=False, ignore_empty=False)(torch.Tensor(seg.astype(np.int32)), torch.Tensor(f_s.astype(np.int32)))
            IoU = MeanIoU(include_background=False, ignore_empty=False)(torch.Tensor(seg.astype(np.int32)), torch.Tensor(f_s.astype(np.int32)))
            IoU_ = MeanIoU(include_background=False, ignore_empty=False)(torch.Tensor(s.astype(np.int32)), torch.Tensor(f_s.astype(np.int32)))
            Dice_ = DiceMetric(include_background=False, ignore_empty=False)(torch.Tensor(s.astype(np.int32)), torch.Tensor(f_s.astype(np.int32)))

            val_res = pd.concat([val_res,
                                  pd.DataFrame.from_records([{'Epoch': '--', 'Task': task, 'ID': case,
                                                              'Dice': np.mean(Dice.cpu().numpy()*100),
                                                              'Dice nm': np.mean(Dice_.cpu().numpy()*100),
                                                              'IoU': np.mean(IoU.cpu().numpy()*100),
                                                              'IoU nm': np.mean(IoU_.cpu().numpy()*100),
                                                              'MSE (moving_to_fixed)': np.mean(MSE_imgs.cpu().numpy())
                                                              }])
                            ], axis=0)

            sitk.WriteImage(sitk.GetImageFromArray(f), os.path.join(out_new, 'fixed_img.nii.gz'))
            sitk.WriteImage(sitk.GetImageFromArray(f_s), os.path.join(out_new, 'fixed_seg.nii.gz'))
            sitk.WriteImage(sitk.GetImageFromArray(moving), os.path.join(out_new, 'moved_img.nii.gz'))
            sitk.WriteImage(sitk.GetImageFromArray(seg), os.path.join(out_new, 'moved_seg.nii.gz'))

        val_res.to_csv(os.path.join(outs[0], task, 'predictions', "validation_results.csv"), index=False, sep=',')  # <-- Includes the header