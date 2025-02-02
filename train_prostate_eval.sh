### Generate the train list for the prostate dataset

domain_list=(domain0 domain1 domain2 domain3 domain4 domain5 )

for domain in "${domain_list[@]}"
do
    CUDA_VISIBLE_DEVICES=6 nnUNet_train 3d_fullres nnUNetInterevalTrainerV2 309 $domain \
        -p nnUNetPlansv2.1_pretrained 
done
