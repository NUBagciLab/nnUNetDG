### Generate the train list for the prostate dataset
export nnUNet_raw_data_base=/data/datasets/nnUNetDG/nnUNet_raw_data_base
export nnUNet_preprocessed=/data/datasets/nnUNetDG/nnUNet_preprocessed
export RESULTS_FOLDER=/data/datasets/nnUNetDG/nnUNet_trained_models

# domain_list=(domain0 domain1 domain2 domain3 domain4 domain5 )
# cuda_list=(0 0 0 1 1 1)
domain_list=(domain0 domain3 )
cuda_list=(0 1 )

# for domain in "${domain_list[@]}"
# do
#     CUDA_VISIBLE_DEVICES=7 nnUNet_train 3d_fullres nnUNetPretrainTrainerV2 309 $domain \
#         -p nnUNetPlansv2.1_pretrained --continue_training
# done
i=0

ulimit -u 10000
for domain in "${domain_list[@]}"
do
    CUDA_VISIBLE_DEVICES=${cuda_list[$i]} nnUNet_def_n_proc=2 nnUNet_train 3d_fullres nnUNetPretrainTrainerV2 309 $domain \
        -p nnUNetPlansv2.1_wpretrained &
    echo $CUDA_VISIBLE_DEVICES
    i=$((i+1))
    wait 60
done
