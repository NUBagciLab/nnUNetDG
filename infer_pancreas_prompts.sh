# CUDA_VISIBLE_DEVICES=6 nnUNet_train 3d_fullres nnUNetPretrainTrainerV2 310 0 \
#     -p nnUNetPlansv2.1_pretrained --fp32
export nnUNet_raw_data_base=/data/datasets/nnUNetDG/nnUNet_raw_data_base
export nnUNet_preprocessed=/data/datasets/nnUNetDG/nnUNet_preprocessed
export RESULTS_FOLDER=/data/datasets/nnUNetDG/nnUNet_trained_models

# CUDA_VISIBLE_DEVICES=0 nnUNet_def_n_proc=2 nnUNet_train 3d_fullres nnUNetPromptTrainerV2 310 0 \
#     -p nnUNetPlansv2.1_prompts --fp32

output_tag="predict_pt5_bbox_points"
base_dir="/data/datasets/nnUNetDG/nnUNet_raw_data_base/nnUNet_raw_data/Task310_PancreasT1DG/out_of_domain"
domain_list=( OutPhaseNU )

for domain in ${domain_list[@]}
do
    echo "Inference on ${domain}"
    input_dir="${base_dir}/${domain}/imagesTr"
    # rm -rf ${base_dir}/${domain}/t1/infer_${output_tag}
    OUTPUT_DIRECTORY="${base_dir}/${domain}/infer_${output_tag}"
    mkdir ${OUTPUT_DIRECTORY}

    CUDA_VISIBLE_DEVICES=1 nnUNet_def_n_proc=2 python nnunet/inference/predict_prompts_resize.py \
        -tr nnUNetPromptTrainerV2 \
        -i ${input_dir} -l ${base_dir}/${domain}/labelsTr -o ${OUTPUT_DIRECTORY} \
        -t 310 -m 3d_fullres -p nnUNetPlansv2.1_ptbottom_5 --overwrite_existing \
        --disable_mixed_precision -chk model_best \
        --use_box_prompt --use_point_prompt --step_size 0.25

    # CUDA_VISIBLE_DEVICES=1 nnUNet_def_n_proc=2 nnUNet_predict \
    #     -tr nnUNetPromptTrainerV2 \
    #     -i ${input_dir} -o ${OUTPUT_DIRECTORY} \
    #     -t 310 -m 3d_fullres -p nnUNetPlansv2.1_ptbottom_5 --overwrite_existing --disable_mixed_precision \
    #     -chk model_best
    CUDA_VISIBLE_DEVICES=7 nnUNet_evaluate_folder -ref ${base_dir}/${domain}/labelsTr -pred ${OUTPUT_DIRECTORY} -l 1
done


