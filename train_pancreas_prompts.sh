# CUDA_VISIBLE_DEVICES=6 nnUNet_train 3d_fullres nnUNetPretrainTrainerV2 310 0 \
#     -p nnUNetPlansv2.1_pretrained --fp32
export nnUNet_raw_data_base=/data/datasets/nnUNetDG/nnUNet_raw_data_base
export nnUNet_preprocessed=/data/datasets/nnUNetDG/nnUNet_preprocessed
export RESULTS_FOLDER=/data/datasets/nnUNetDG/nnUNet_trained_models

# CUDA_VISIBLE_DEVICES=0 nnUNet_def_n_proc=2 nnUNet_train 3d_fullres nnUNetPromptTrainerV2 310 0 \
#     -p nnUNetPlansv2.1_prompts --fp32 --continue_training

# CUDA_VISIBLE_DEVICES=1 nnUNet_def_n_proc=2 nnUNet_train 3d_fullres nnUNetPromptTrainerV2 310 0 \
#     -p nnUNetPlansv2.1_prompttune --fp32 --continue_training

CUDA_VISIBLE_DEVICES=2 nnUNet_def_n_proc=2 nnUNet_train 3d_fullres nnUNetPromptTrainerV2 310 0 \
    -p nnUNetPlansv2.1_fptbottom_5 --fp32 --continue_training
