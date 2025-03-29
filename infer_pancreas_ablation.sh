# CUDA_VISIBLE_DEVICES=6 nnUNet_train 3d_fullres nnUNetPretrainTrainerV2 310 0 \
#     -p nnUNetPlansv2.1_pretrained --fp32
export nnUNet_raw_data_base=/data/datasets/nnUNetDG/nnUNet_raw_data_base
export nnUNet_preprocessed=/data/datasets/nnUNetDG/nnUNet_preprocessed
export RESULTS_FOLDER=/data/datasets/nnUNetDG/nnUNet_trained_models

# CUDA_VISIBLE_DEVICES=0 nnUNet_def_n_proc=2 nnUNet_train 3d_fullres nnUNetPromptTrainerV2 310 0 \
#     -p nnUNetPlansv2.1_prompts --fp32

# base_dir="/data/datasets/nnUNetDG/nnUNet_raw_data_base/nnUNet_raw_data/Task310_PancreasT1DG/out_of_domain"
base_dir="/data/datasets/nnUNetDG/nnUNet_raw_data_base/nnUNet_raw_data/Task310_PancreasT1DG/in_domain"
# domain_list=(EMC NU IU OutPhaseEMC OutPhaseNU )
# domain_list=(OutPhaseNU_total )
domain_list=(MCF NYU )

name_list=(nnUNetPlansv2.1_ptbottom_1 \
           nnUNetPlansv2.1_ptbottom_2 \
           nnUNetPlansv2.1_ptbottom_3 \
           nnUNetPlansv2.1_ptbottom_4 \
           nnUNetPlansv2.1_ptbottom_5 \
           nnUNetPlansv2.1_ptbottom_6 )

for name in ${name_list[@]}
do
    for domain in ${domain_list[@]}
    do
        echo "Use ${name} to as plan"
        echo "Inference on ${domain}"
        input_dir="${base_dir}/${domain}/imagesTr"
        # rm -rf ${base_dir}/${domain}/t1/infer_${output_tag}
        OUTPUT_DIRECTORY="${base_dir}/${domain}/infer_${name}_woprompt"
        mkdir ${OUTPUT_DIRECTORY}

        CUDA_VISIBLE_DEVICES=1 nnUNet_def_n_proc=2 python nnunet/inference/predict_prompts.py \
            -tr nnUNetPromptTrainerV2 \
            -i ${input_dir} -l ${base_dir}/${domain}/labelsTr -o ${OUTPUT_DIRECTORY} \
            -t 310 -m 3d_fullres -p ${name} --overwrite_existing \
            --disable_mixed_precision -chk model_best
            # --use_box_prompt --use_point_prompt

        CUDA_VISIBLE_DEVICES=1 nnUNet_evaluate_folder -ref ${base_dir}/${domain}/labelsTr -pred ${OUTPUT_DIRECTORY} -l 1

        OUTPUT_DIRECTORY="${base_dir}/${domain}/infer_${name}_wbox"
        mkdir ${OUTPUT_DIRECTORY}

        CUDA_VISIBLE_DEVICES=1 nnUNet_def_n_proc=2 python nnunet/inference/predict_prompts.py \
            -tr nnUNetPromptTrainerV2 \
            -i ${input_dir} -l ${base_dir}/${domain}/labelsTr -o ${OUTPUT_DIRECTORY} \
            -t 310 -m 3d_fullres -p ${name} --overwrite_existing \
            --disable_mixed_precision -chk model_best \
            --use_box_prompt 

        CUDA_VISIBLE_DEVICES=1 nnUNet_evaluate_folder -ref ${base_dir}/${domain}/labelsTr -pred ${OUTPUT_DIRECTORY} -l 1
        
        OUTPUT_DIRECTORY="${base_dir}/${domain}/infer_${name}_wpoint"
        mkdir ${OUTPUT_DIRECTORY}

        CUDA_VISIBLE_DEVICES=1 nnUNet_def_n_proc=2 python nnunet/inference/predict_prompts.py \
            -tr nnUNetPromptTrainerV2 \
            -i ${input_dir} -l ${base_dir}/${domain}/labelsTr -o ${OUTPUT_DIRECTORY} \
            -t 310 -m 3d_fullres -p ${name} --overwrite_existing \
            --disable_mixed_precision -chk model_best \
            --use_point_prompt

        CUDA_VISIBLE_DEVICES=1 nnUNet_evaluate_folder -ref ${base_dir}/${domain}/labelsTr -pred ${OUTPUT_DIRECTORY} -l 1
        
        OUTPUT_DIRECTORY="${base_dir}/${domain}/infer_${name}_wbox_point"
        mkdir ${OUTPUT_DIRECTORY}

        CUDA_VISIBLE_DEVICES=1 nnUNet_def_n_proc=2 python nnunet/inference/predict_prompts.py \
            -tr nnUNetPromptTrainerV2 \
            -i ${input_dir} -l ${base_dir}/${domain}/labelsTr -o ${OUTPUT_DIRECTORY} \
            -t 310 -m 3d_fullres -p ${name} --overwrite_existing \
            --disable_mixed_precision -chk model_best \
            --use_point_prompt --use_box_prompt 

        CUDA_VISIBLE_DEVICES=1 nnUNet_evaluate_folder -ref ${base_dir}/${domain}/labelsTr -pred ${OUTPUT_DIRECTORY} -l 1
        
    done
done


