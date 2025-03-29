# CUDA_VISIBLE_DEVICES=6 nnUNet_train 3d_fullres nnUNetPretrainTrainerV2 310 0 \
#     -p nnUNetPlansv2.1_pretrained --fp32
export nnUNet_raw_data_base=/data/datasets/nnUNetDG/nnUNet_raw_data_base
export nnUNet_preprocessed=/data/datasets/nnUNetDG/nnUNet_preprocessed
export RESULTS_FOLDER=/data/datasets/nnUNetDG/nnUNet_trained_models

# for domain in "${domain_list[@]}"
# do
#     CUDA_VISIBLE_DEVICES=7 nnUNet_train 3d_fullres nnUNetPretrainTrainerV2 309 $domain \
#         -p nnUNetPlansv2.1_pretrained --continue_training ${cuda_list[$i]}
# done
i=0
# name_list=(nnUNetPlansv2.1_ptbottom_4 \
#            nnUNetPlansv2.1_ptbottom_5 \
#            nnUNetPlansv2.1_ptbottom_6 )
name_list=(nnUNetPlansv2.1_fptbottom_6 )

for name in "${name_list[@]}"
do
    echo ${name_list[$i]} 
    CUDA_VISIBLE_DEVICES=2 nnUNet_def_n_proc=2 nnUNet_train 3d_fullres nnUNetPromptTrainerV2 310 0 \
        -p ${name_list[$i]} --fp32
    i=$((i+1))
done

