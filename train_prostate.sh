### Generate the train list for the prostate dataset

domain_list=(domain1 domain2 domain3 domain4 domain5 )

for domain in "${domain_list[@]}"
do
    CUDA_VISIBLE_DEVICES=7 nnUNet_train 3d_fullres nnUNetPretrainTrainerV2 309 $domain \
        -p nnUNetPlansv2.1_pretrained 
done
