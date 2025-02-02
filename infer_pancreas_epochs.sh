# domain_list=( OutPhaseEMC OutPhaseNU )
# domain_list=( EMC NU IU OutPhaseEMC OutPhaseNU)
# domain_list=(EMC NU IU OutPhaseNU )
domain_list=(MCF NYU )
# output_tag="baseline"
# base_dir="/data/datasets/nnUNetDG/nnUNet_raw_data_base/nnUNet_raw_data/Task310_PancreasT1DG/out_of_domain"
export nnUNet_raw_data_base=/data/datasets/nnUNetDG/nnUNet_raw_data_base
export nnUNet_preprocessed=/data/datasets/nnUNetDG/nnUNet_preprocessed
export RESULTS_FOLDER=/data/datasets/nnUNetDG/nnUNet_trained_models

model_list=(model_ep_100 model_ep_200 model_ep_300 model_ep_400 model_ep_500 \
            model_ep_600 model_ep_700 model_ep_800 model_ep_900 model_ep_1000)
base_dir="/data/datasets/nnUNetDG/nnUNet_raw_data_base/nnUNet_raw_data/Task310_PancreasT1DG/in_domain"

for model in ${model_list[@]}
do
    for domain in ${domain_list[@]}
    do
        output_tag="${model}"
        echo "Inference on ${output_tag}"
        echo "Inference on ${domain}"
        input_dir="${base_dir}/${domain}/imagesTr"
        rm -rf ${base_dir}/${domain}/t1/infer_${output_tag}
        mkdir ${base_dir}/${domain}/infer_${output_tag}
        OUTPUT_DIRECTORY="${base_dir}/${domain}/infer_${output_tag}"
        CUDA_VISIBLE_DEVICES=7 nnUNet_predict -tr nnUNetStandardTrainerV2 \
        -i ${input_dir} -o ${OUTPUT_DIRECTORY} -t 310 -m 3d_fullres --folds 0 \
        -p nnUNetPlansv2.1_pretrained -chk ${model}
        CUDA_VISIBLE_DEVICES=7 nnUNet_evaluate_folder -ref ${base_dir}/${domain}/labelsTr -pred ${OUTPUT_DIRECTORY} -l 1
    done
done
