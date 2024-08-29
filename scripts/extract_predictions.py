import sys, os, logging, time
import numpy as np
from tqdm import tqdm
from importlib import reload
from scripts import register

ins = [
        # '/media/aranem_locale/AR_subs_exps/MIDL_2024_NCA-Morph/MIDL_2024_raw_data/mapped_rigid/nnUNet_ready_with_ts_folder_rigid_aligned/Task210_OASIS/imagesTs'
        # '/local/scratch/aranem/BMVC_2024/BMVC_2024_raw_data/mapped_rigid/nnUNet_ready_with_ts_folder_rigid_aligned/Task210_OASIS/imagesTs'
        # '/local/scratch/aranem/BMVC_2024/BMVC_2024_raw_data/mapped_rigid/nnUNet_ready_with_ts_folder_rigid_aligned/Task211_Prostate/imagesTs'
        # '/local/scratch/aranem/BMVC_2024/BMVC_2024_raw_data/mapped_rigid/nnUNet_ready_with_ts_folder_rigid_aligned/Task212_Hippocampus/imagesTs'
        # '/local/scratch/aranem/BMVC_2024/BMVC_2024_raw_data/mapped_rigid/nnUNet_ready_with_ts_folder_rigid_aligned/Task213_Cardiac/imagesTs'
      ]

models = [
        # '/media/aranem_locale/AR_subs_exps/MIDL_2024_NCA-Morph/MIDL_2024_trained_models/NCA_Morph/nca_morph_torch_250_210_ncc_kernel_7_10_32_128_direct_flow/0250.pt',
        # '/media/aranem_locale/AR_subs_exps/MIDL_2024_NCA-Morph/MIDL_2024_trained_models/TransMorph/transmorph_torch_250_210_ncc_ce_default/0250.pt',
        # '/media/aranem_locale/AR_subs_exps/MIDL_2024_NCA-Morph/MIDL_2024_trained_models/VoxelMorph/vxm_torch_250_210_ncc_ce/0250.pt'
        #   '/local/scratch/aranem/BMVC_2024/BMVC_2024_trained_models/ViTVNet/vitvnet_torch_250_212_ncc_ce/0250.pt',
        #   '/local/scratch/aranem/BMVC_2024/BMVC_2024_trained_models/TransMorph/transmorph_torch_250_212_ncc_ce_tiny/0250.pt',
        #   '/local/scratch/aranem/BMVC_2024/BMVC_2024_trained_models/VoxelMorph/vxm_torch_250_212_ncc_ce/0100.pt',
        #   '/local/scratch/aranem/BMVC_2024/BMVC_2024_trained_models/NCA_Morph/vxm_nca_torch_250_212_ncc_kernel_7_30_16_64/0100.pt',
        #   '/local/scratch/aranem/BMVC_2024/BMVC_2024_trained_models/NCA_Morph/vxm_nca_torch_250_212_ncc_kernel_7_7_10/0250.pt',
        #   '/local/scratch/aranem/BMVC_2024/BMVC_2024_trained_models/NCA_Morph/vxm_nca_torch_250_213_ncc_kernel_7_7_10/0250.pt',
        #   '/local/scratch/aranem/BMVC_2024/BMVC_2024_trained_models/ViTVNet/vitvnet_torch_250_210_ncc_ce/0250.pt',
        #   '/local/scratch/aranem/BMVC_2024/BMVC_2024_trained_models/ViTVNet/vitvnet_torch_250_211_ncc_ce/0250.pt',
        #   '/local/scratch/aranem/BMVC_2024/BMVC_2024_trained_models/TransMorph/transmorph_torch_250_210_ncc_ce_default/0250.pt',
        #   '/local/scratch/aranem/BMVC_2024/BMVC_2024_trained_models/TransMorph/transmorph_torch_250_210_ncc_ce_tiny/0250.pt',
        #   '/local/scratch/aranem/BMVC_2024/BMVC_2024_trained_models/TransMorph/transmorph_torch_250_211_ncc_ce_tiny/0250.pt',
        #   '/local/scratch/aranem/BMVC_2024/BMVC_2024_trained_models/TransMorph/transmorph_torch_250_210_ncc_ce_small/0250.pt',
        #   '/local/scratch/aranem/BMVC_2024/BMVC_2024_trained_models/TransMorph/transmorph_torch_250_210_ncc_ce_large/0250.pt',
        #   '/local/scratch/aranem/BMVC_2024/BMVC_2024_trained_models/TransMorph/transmorph_torch_250_211_ncc_ce_large/0250.pt',
        #   '/local/scratch/aranem/BMVC_2024/BMVC_2024_trained_models/VoxelMorph/vxm_torch_250_210_ncc_ce/0250.pt',
        #   '/local/scratch/aranem/BMVC_2024/BMVC_2024_trained_models/VoxelMorph/vxm_torch_250_211_ncc_ce/0250.pt',
        #   '/local/scratch/aranem/BMVC_2024/BMVC_2024_trained_models/NCA_Morph/vxm_nca_torch_250_210_ncc_kernel_3_3_30/0250.pt',
        #   '/local/scratch/aranem/BMVC_2024/BMVC_2024_trained_models/NCA_Morph/vxm_nca_torch_250_210_ncc_kernel_5_5_30/0250.pt',
        #   '/local/scratch/aranem/BMVC_2024/BMVC_2024_trained_models/NCA_Morph/vxm_nca_torch_250_210_ncc_kernel_7_7_30/0250.pt',
        #   '/local/scratch/aranem/BMVC_2024/BMVC_2024_trained_models/NCA_Morph/vxm_nca_torch_250_210_ncc_kernel_9_9_30/0250.pt',
        #   '/local/scratch/aranem/BMVC_2024/BMVC_2024_trained_models/NCA_Morph/vxm_nca_torch_250_210_ncc_kernel_7_7_5/0250.pt',
        #   '/local/scratch/aranem/BMVC_2024/BMVC_2024_trained_models/NCA_Morph/vxm_nca_torch_250_210_ncc_kernel_7_7_10/0250.pt',
        #   '/local/scratch/aranem/BMVC_2024/BMVC_2024_trained_models/NCA_Morph/vxm_nca_torch_250_210_ncc_kernel_7_7_50/0250.pt',
        #   '/local/scratch/aranem/BMVC_2024/BMVC_2024_trained_models/NCA_Morph/vxm_nca_torch_250_210_ncc_kernel_7_7_90/0250.pt',
        #   '/local/scratch/aranem/BMVC_2024/BMVC_2024_trained_models/NCA_Morph/vxm_nca_torch_250_210_ncc_kernel_7_7_10_direct_flow/0250.pt',
        #   '/local/scratch/aranem/BMVC_2024/BMVC_2024_trained_models/NCA_Morph/vxm_nca_torch_250_210_ncc_kernel_7_7_30_direct_flow/0250.pt',
        #   '/local/scratch/aranem/BMVC_2024/BMVC_2024_trained_models/NCA_Morph/vxm_nca_torch_250_210_ncc_kernel_7_7_10_direct_flow_fire_rate_25/0250.pt',
        #   '/local/scratch/aranem/BMVC_2024/BMVC_2024_trained_models/NCA_Morph/vxm_nca_torch_250_210_ncc_kernel_7_7_10_direct_flow_fire_rate_50/0250.pt',
        #   '/local/scratch/aranem/BMVC_2024/BMVC_2024_trained_models/NCA_Morph/vxm_nca_torch_250_210_ncc_kernel_7_7_10_direct_flow_fire_rate_75/0250.pt',
        #   '/local/scratch/aranem/BMVC_2024/BMVC_2024_trained_models/NCA_Morph/vxm_nca_torch_250_210_ncc_kernel_7_7_10_direct_flow_fire_rate_100/0250.pt',
        #   '/local/scratch/aranem/BMVC_2024/BMVC_2024_trained_models/NCA_Morph/vxm_nca_torch_250_210_ncc_kernel_7_7_10_fire_rate_25/0250.pt',
        #   '/local/scratch/aranem/BMVC_2024/BMVC_2024_trained_models/NCA_Morph/vxm_nca_torch_250_210_ncc_kernel_7_7_10_fire_rate_50/0250.pt',
        #   '/local/scratch/aranem/BMVC_2024/BMVC_2024_trained_models/NCA_Morph/vxm_nca_torch_250_210_ncc_kernel_7_7_10_fire_rate_75/0250.pt',
        #   '/local/scratch/aranem/BMVC_2024/BMVC_2024_trained_models/NCA_Morph/vxm_nca_torch_250_210_ncc_kernel_7_7_10_fire_rate_100/0250.pt',
        # -- Augmentations -- #
        #   '/local/scratch/aranem/BMVC_2024/BMVC_2024_trained_models/VoxelMorph/vxm_torch_250_210_ncc_ce_bias_level_1/0250.pt',
        #   '/local/scratch/aranem/BMVC_2024/BMVC_2024_trained_models/VoxelMorph/vxm_torch_250_210_ncc_ce_bias_level_2/0250.pt',
        #   '/local/scratch/aranem/BMVC_2024/BMVC_2024_trained_models/VoxelMorph/vxm_torch_250_210_ncc_ce_bias_level_3/0250.pt',
        #   '/local/scratch/aranem/BMVC_2024/BMVC_2024_trained_models/VoxelMorph/vxm_torch_250_210_ncc_ce_ghosting_level_1/0250.pt',
        #   '/local/scratch/aranem/BMVC_2024/BMVC_2024_trained_models/VoxelMorph/vxm_torch_250_210_ncc_ce_ghosting_level_2/0250.pt',
        #   '/local/scratch/aranem/BMVC_2024/BMVC_2024_trained_models/VoxelMorph/vxm_torch_250_210_ncc_ce_ghosting_level_3/0250.pt',
        #   '/local/scratch/aranem/BMVC_2024/BMVC_2024_trained_models/VoxelMorph/vxm_torch_250_210_ncc_ce_spike_level_1/0250.pt',
        #   '/local/scratch/aranem/BMVC_2024/BMVC_2024_trained_models/VoxelMorph/vxm_torch_250_210_ncc_ce_spike_level_2/0250.pt',
        #   '/local/scratch/aranem/BMVC_2024/BMVC_2024_trained_models/VoxelMorph/vxm_torch_250_210_ncc_ce_spike_level_3/0250.pt',
        #   '/local/scratch/aranem/BMVC_2024/BMVC_2024_trained_models/NCA_Morph/vxm_nca_torch_250_210_ncc_kernel_7_7_10_bias_level_1/0250.pt',
        #   '/local/scratch/aranem/BMVC_2024/BMVC_2024_trained_models/NCA_Morph/vxm_nca_torch_250_210_ncc_kernel_7_7_10_bias_level_2/0250.pt',
        #   '/local/scratch/aranem/BMVC_2024/BMVC_2024_trained_models/NCA_Morph/vxm_nca_torch_250_210_ncc_kernel_7_7_10_bias_level_3/0250.pt',
        #   '/local/scratch/aranem/BMVC_2024/BMVC_2024_trained_models/NCA_Morph/vxm_nca_torch_250_210_ncc_kernel_7_7_10_ghosting_level_1/0250.pt',
        #   '/local/scratch/aranem/BMVC_2024/BMVC_2024_trained_models/NCA_Morph/vxm_nca_torch_250_210_ncc_kernel_7_7_10_ghosting_level_2/0250.pt',
        #   '/local/scratch/aranem/BMVC_2024/BMVC_2024_trained_models/NCA_Morph/vxm_nca_torch_250_210_ncc_kernel_7_7_10_ghosting_level_3/0250.pt',
        #   '/local/scratch/aranem/BMVC_2024/BMVC_2024_trained_models/NCA_Morph/vxm_nca_torch_250_210_ncc_kernel_7_7_10_spike_level_1/0250.pt',
        #   '/local/scratch/aranem/BMVC_2024/BMVC_2024_trained_models/NCA_Morph/vxm_nca_torch_250_210_ncc_kernel_7_7_10_spike_level_2/0250.pt',
        #   '/local/scratch/aranem/BMVC_2024/BMVC_2024_trained_models/NCA_Morph/vxm_nca_torch_250_210_ncc_kernel_7_7_10_spike_level_3/0250.pt',
          ]

# -- Example inputs for reference -- #
# --moving /local/scratch/aranem/BMVC_2024/BMVC_2024_raw_data/mapped_rigid/nnUNet_ready_with_ts_folder_rigid_aligned/Task/imagesTr/Case08_0001.nii.gz
# --fixed /local/scratch/aranem/BMVC_2024/BMVC_2024_raw_data/mapped_rigid/nnUNet_ready_with_ts_folder_rigid_aligned/Task/imagesTr/Case08_0000.nii.gz
# --moving_seg /local/scratch/aranem/BMVC_2024/BMVC_2024_raw_data/mapped_rigid/nnUNet_ready_with_ts_folder_rigid_aligned/Task/labelsTr/Case08_0001.nii.gz
# --model home/aranem_locale/Desktop/BMVC_2024/BMVC_2024_trained_models/NCA_Morph/vxm_torch_250_110_ncc_dice/0200.pt
# --out home/aranem_locale/Desktop/BMVC_2024/BMVC_2024_trained_models/NCA_Morph/vxm_torch_250_110_ncc_dice/eval/Case08/
# --gt /local/scratch/aranem/BMVC_2024/BMVC_2024_raw_data/mapped_rigid/nnUNet_ready_with_ts_folder_rigid_aligned/Task/labelsTr/Case08_0000.nii.gz

    
# Extract predictions
for model in models:
    vitvnet, transmorph, nca, nicetrans, seg = False, False, False, False, False
    if 'unet' in model:
        seg = True
    if 'nca' in model:
        nca = True
        seg = False
    if 'transmorph' in model:
        transmorph = True
        seg = False
    if 'vitvnet' in model:
        vitvnet = True
        seg = False
    if 'nicetrans' in model:
        nicetrans = True
        seg = False
    
    # set logger to also log the messages
    inference_step_time = []

    for inp in ins:
        res_Dice, res_IoU = [], []
        res_Dice_, res_IoU_ = [], []
        res_MSE = []
        print(f"Creating predictions with {model.split(os.sep)[-2]} for {inp.split(os.sep)[-2]}:")
        
        os.makedirs(os.path.join(os.path.sep, *model.split(os.path.sep)[:-1], 'predictions', inp.split(os.sep)[-2]), exist_ok=True)
        logging.basicConfig(filename = os.path.join(os.path.sep, *model.split(os.path.sep)[:-1], 'predictions', inp.split(os.sep)[-2], "inference_log.txt"), 
                            format = '%(asctime)s %(message)s') 
        logger = logging.getLogger() 
        logger.setLevel(logging.DEBUG)
        logger.debug(f"Creating predictions with {model.split(os.sep)[-2]} for {inp.split(os.sep)[-2]}:")
        
        out_ = os.path.join(os.path.sep, *model.split(os.path.sep)[:-1], 'predictions', inp.split(os.sep)[-2])
        cases = [x[:-12] for x in os.listdir(inp) if '._' not in x and '.json' not in x and 'DS_Store' not in x]
        cases = np.unique(cases)
        
        for case in tqdm(cases):
            step_start_time = time.time()
            fixed = os.path.join(inp, case+'_0000.nii.gz')
            moving = os.path.join(inp, case+'_0001.nii.gz')
            moving_seg = os.path.join(inp.replace('imagesTs', 'labelsTs'), case+'_0001.nii.gz')
            gt = os.path.join(inp.replace('imagesTs', 'labelsTs'), case+'_0000.nii.gz')
            out = os.path.join(out_, case)
            os.makedirs(out, exist_ok=True)

            # -- Build up arguments and do registration -- #
            args = [sys.argv[0], '--model']
            args += [model, '--fixed']
            args += [fixed, '--moving']
            args += [moving, '--moving_seg']
            args += [moving_seg, '--gt']
            args += [gt, '--out']
            args += [out]
            if seg:
                args += ['--seg']
            if nca:
                args += ['--nca']
            if transmorph:
                args += ['--transmorph']
            if vitvnet:
                args += ['--vitvnet']
            if nicetrans:
                args += ['--nicetrans']
            sys.argv = args
            register = reload(register)   # So the log files can be updated as well
            Dice, IoU, Dice_, IoU_, MSE = register.register()
            inference_step_time.append(time.time() - step_start_time)
            res_Dice.append(Dice)
            res_IoU.append(IoU)
            res_Dice_.append(Dice_)
            res_IoU_.append(IoU_)
            res_MSE.append(MSE)
        
        inference_info = 'Inference'
        time_img_info = '%.4f sec/img' % np.mean(inference_step_time)
        time_inference_info = '%.4f sec/total' % np.sum(inference_step_time)
        img_info = '{} images in total'.format(len(cases))
        logger.debug(' - '.join((inference_info, time_img_info, time_inference_info, img_info)))
        logger.debug(f"Performance of model {model.split(os.sep)[-2]} for {inp.split(os.sep)[-2]} (Dice -- IoU -- MSE | Dice nm -- IoU nm): {np.mean(res_Dice):.2f}% -- {np.mean(res_IoU):.2f}% -- {np.mean(res_MSE):.2f}% | {np.mean(res_Dice_):.2f}% -- {np.mean(res_IoU_):.2f}%")