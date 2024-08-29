#!/usr/bin/env python

# VOXELMORPH 210, 211, 212

"""
Script to run multiple MIDL_2024_trained_models in a sequence using a Telegram Bot to get update messages.
This can be used to trigger one time a list of MIDL_2024_trained_models without executing them one by one.
It also checks if an experiment is finished and if so, it will skip it, else it continues with the training.
"""
import sys, os
from scripts import train
from importlib import reload

# -- Set configurations manually -- #
device = 0
nr_epochs = 250
mappings = {'210': 'Task210_OASIS', '211': "Task211_Prostate", '212': "Task212_Hippocampus", '213': "Task213_Cardiac"}
train_on = [['211']]

continue_ = False
finished = False
continue_with_epoch = 0


# train.py --img-list /local/scratch/aranem/MIDL_2024/MIDL_2024_trained_models/VoxelMorph/train_list.txt 
#          --seg-list /local/scratch/aranem/MIDL_2024/MIDL_2024_trained_models/VoxelMorph/train_seg.txt
#          --model-dir /local/scratch/aranem/MIDL_2024/MIDL_2024_trained_models/VoxelMorph/vxm_torch_250_110_111_112_113_ncc_ce
#          --load-model /local/scratch/aranem/MIDL_2024/MIDL_2024_trained_models/VoxelMorph/vxm_torch_250_110_111_112_113_ncc_ce/0250.pt
#          --initial-epoch 10
#          --gpu 0
#          --epochs 250
#          --seg

# -- Train based on the configurations -- #
for tasks in train_on:
    trained_list = []
    for task in tasks:
        prev_mod_built = '_'.join(trained_list)
        trained_list.append(task)
        built_ts = '_'.join(trained_list)
        img_list = f'/local/scratch/aranem/MIDL_2024/MIDL_2024_trained_models/list_files/{mappings[task]}/train_list.txt'
        seg_list = f'/local/scratch/aranem/MIDL_2024/MIDL_2024_trained_models/list_files/{mappings[task]}/train_seg.txt'
        out_folder = f'/local/scratch/aranem/MIDL_2024/MIDL_2024_trained_models/VoxelMorph/vxm_torch_{nr_epochs}_{built_ts}_ncc_ce'

        # -- Check if it is already trained or not -- #
        if os.path.exists(out_folder):
            # -- Started training on, so restore if more than one checkpoint -- #
            chks = [x for x in os.listdir(out_folder) if '.pt' in x]
            if len(chks) <= 1:  # Only 0000.pt in the list
                if len(trained_list) > 1: # <-- We still need load_model here
                    prev_model = out_folder.replace(built_ts, prev_mod_built)
                    continue_, finished, continue_with_epoch = True, True, 0
                    load_model = os.path.join(prev_model, '%04d.pt' % nr_epochs)    # <-- Should exist!
                else:
                    continue_, finished, continue_with_epoch = False, False, 0
            else:
                chks.sort()
                chkp = chks[-1]
                if str(nr_epochs) in chkp:
                    continue_, finished, continue_with_epoch = False, False, 0
                    continue    # <-- Finished with training for this task
                continue_, finished, continue_with_epoch = True, False, int(chkp.split('.pt')[0][1:])
                load_model = os.path.join(out_folder, '%04d.pt' % continue_with_epoch)

        elif len(trained_list) > 1: # <-- We still need load_model here
            prev_model = out_folder.replace(built_ts, prev_mod_built)
            continue_, finished, continue_with_epoch = True, True, 0
            load_model = os.path.join(prev_model, '%04d.pt' % nr_epochs)    # <-- Should exist!

        # -- Build up arguments -- #
        args = [sys.argv[0], '--img-list']
        args += [img_list]
        args += ['--seg-list', seg_list]
        args += ['--model-dir', out_folder]
        if continue_:
            args += ['--load-model', load_model]
            if not finished:
                args += ['--initial-epoch', str(continue_with_epoch)]
        args += ['--gpu', str(device)]
        args += ['--epochs', str(nr_epochs)]

        # -- Train -- #
        sys.argv = args
        train = reload(train)
        train.train()