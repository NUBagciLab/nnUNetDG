#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import argparse
from copy import deepcopy
from typing import Tuple, Union, List

import numpy as np

from batchgenerators.augmentations.utils import resize_segmentation, resize_multichannel_image
from nnunet.inference.segmentation_export import save_segmentation_nifti_from_softmax, save_segmentation_nifti
from batchgenerators.utilities.file_and_folder_operations import *
import sys
if 'win' in sys.platform:
    #fix for windows platform
    import pathos
    Process = pathos.helpers.mp.Process
    Queue = pathos.helpers.mp.Queue
else:
    from multiprocessing import Process, Queue
import torch
import random
import SimpleITK as sitk
import shutil
from multiprocessing import Pool
from skimage.transform import resize
from nnunet.postprocessing.connected_components import load_remove_save, load_postprocessing
from nnunet.training.model_restore import load_model_and_checkpoint_files
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.utilities.one_hot_encoding import to_one_hot

from nnunet.paths import default_plans_identifier, network_training_output_dir, default_cascade_trainer, default_trainer
from batchgenerators.utilities.file_and_folder_operations import join, isdir, save_json
from batchgenerators.augmentations.utils import pad_nd_image
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name
from time import time



def preprocess_save_to_queue(preprocess_fn, q, list_of_lists, seg_of_lists, output_files, segs_from_prev_stage, classes,
                             transpose_forward):
    # suppress output
    # sys.stdout = open(os.devnull, 'w')

    errors_in = []
    for i, l in enumerate(list_of_lists):
        try:
            output_file = output_files[i]
            seg_file = seg_of_lists[i]
            print("preprocessing", output_file)
            d, s, dct = preprocess_fn(l, seg_file)
            # print(output_file, dct)
            """There is a problem with python process communication that prevents us from communicating objects 
            larger than 2 GB between processes (basically when the length of the pickle string that will be sent is 
            communicated by the multiprocessing.Pipe object then the placeholder (I think) does not allow for long 
            enough strings (lol). This could be fixed by changing i to l (for long) but that would require manually 
            patching system python code. We circumvent that problem here by saving softmax_pred to a npy file that will 
            then be read (and finally deleted) by the Process. save_segmentation_nifti_from_softmax can take either 
            filename or np.ndarray and will handle this automatically"""
            print(d.shape)
            if np.prod(d.shape) > (2e9 / 4 * 0.85):  # *0.85 just to be save, 4 because float32 is 4 bytes
                print(
                    "This output is too large for python process-process communication. "
                    "Saving output temporarily to disk")
                np.save(output_file[:-7] + ".npy", d)
                d = output_file[:-7] + ".npy"
            q.put((output_file, (d, s, dct)))
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except Exception as e:
            print("error in", l)
            print(e)
    q.put("end")
    if len(errors_in) > 0:
        print("There were some errors in the following cases:", errors_in)
        print("These cases were ignored.")
    else:
        print("This worker has ended successfully, no errors to report")
    # restore output
    # sys.stdout = sys.__stdout__


def preprocess_multithreaded(trainer, list_of_lists, seg_of_lists, output_files, num_processes=2, segs_from_prev_stage=None):
    if segs_from_prev_stage is None:
        segs_from_prev_stage = [None] * len(list_of_lists)

    num_processes = min(len(list_of_lists), num_processes)

    classes = list(range(1, trainer.num_classes))
    assert isinstance(trainer, nnUNetTrainer)
    q = Queue(1)
    processes = []
    for i in range(num_processes):
        pr = Process(target=preprocess_save_to_queue, args=(trainer.preprocess_patient, q,
                                                            list_of_lists[i::num_processes],
                                                            seg_of_lists[i::num_processes],
                                                            output_files[i::num_processes],
                                                            segs_from_prev_stage[i::num_processes],
                                                            classes, trainer.plans['transpose_forward']))
        pr.start()
        processes.append(pr)

    try:
        end_ctr = 0
        while end_ctr != num_processes:
            item = q.get()
            if item == "end":
                end_ctr += 1
                continue
            else:
                yield item

    finally:
        for p in processes:
            if p.is_alive():
                p.terminate()  # this should not happen but better safe than sorry right
            p.join()

        q.close()

def generate_box_np(pred_pre, bbox_shift=None):
    ### Generate same version like the torch version for numpy array
    meaning_post_label = pred_pre # [h, w, d]
    ones_idx = np.where(meaning_post_label > 0.5)
    if all(len(dim) == 0 for dim in ones_idx):
        bboxes = np.array([-1,-1,-1,-1,-1,-1])
        # print(bboxes, bboxes.shape)
        return bboxes
    
    min_coords = [np.min(dim) for dim in ones_idx]    # [x_min, y_min, z_min]
    max_coords = [np.max(dim) for dim in ones_idx]    # [x_max, y_max, z_max]

    if bbox_shift is None:
        corner_min = []
        corner_max = []
        shape = meaning_post_label.shape
        for coor in min_coords:
            coor_ = max(0, coor)
            corner_min.append(coor_)
        for idx, coor in enumerate(max_coords):
            coor_ = min(shape[idx], coor)
            corner_max.append(coor_)
        corner_min = np.array(corner_min)
        corner_max = np.array(corner_max)
        return np.concatenate((corner_min, corner_max), axis=0)
    else:
        # add perturbation to bounding box coordinates
        corner_min = []
        corner_max = []
        shape = meaning_post_label.shape
        for coor in min_coords:
            coor_ = max(0, coor + random.randint(-bbox_shift, bbox_shift))
            corner_min.append(coor_)
        for idx, coor in enumerate(max_coords):
            coor_ = min(shape[idx], coor + random.randint(-bbox_shift, bbox_shift))
            corner_max.append(coor_)
        corner_min = np.array(corner_min)
        corner_max = np.array(corner_max)
        return np.concatenate((corner_min, corner_max), axis=0)


def select_points_np(preds, num_positive_extra=4, num_negative_extra=0, fix_extra_point_num=None):
    spacial_dim = 3
    points = np.zeros((0, 3))
    labels = np.zeros((0))
    pos_thred = 0.9
    neg_thred = 0.1
    
    # get pos/net indices
    positive_indices = np.where(preds > pos_thred) # ([pos x], [pos y], [pos z])
    negative_indices = np.where(preds < neg_thred)

    ones_idx = np.where(preds > pos_thred)
    if all(len(tmp) == 0 for tmp in ones_idx):
        # all neg
        num_positive_extra = 0
        selected_positive_point = np.array([-1,-1,-1]).reshape(1, 3)
        points = np.concatenate((points, selected_positive_point), axis=0)
        labels = np.concatenate((labels, np.array([-1]).reshape(1)), axis=0)
    else:
        # random select a pos point
        random_idx = np.random.randint(0, len(positive_indices[0]), (1,))
        selected_positive_point = np.array([positive_indices[i][random_idx] for i in range(spacial_dim)]).reshape(1, 3)
        points = np.concatenate((points, selected_positive_point), axis=0)
        labels = np.concatenate((labels, np.ones((1))), axis=0)

    if num_positive_extra > 0:
        pos_idx_list = np.random.permutation(len(positive_indices[0]))[:num_positive_extra]
        extra_positive_points = []
        for pos_idx in pos_idx_list:
            extra_positive_points.append([positive_indices[i][pos_idx] for i in range(spacial_dim)])
        extra_positive_points = np.array(extra_positive_points).reshape(-1, 3)
        points = np.concatenate((points, extra_positive_points), axis=0)
        labels = np.concatenate((labels, np.ones((extra_positive_points.shape[0]))))

    if num_negative_extra > 0:
        neg_idx_list = np.random.permutation(len(negative_indices[0]))[:num_negative_extra]
        extra_negative_points = []
        for neg_idx in neg_idx_list:
            extra_negative_points.append([negative_indices[i][neg_idx] for i in range(spacial_dim)])
        extra_negative_points = np.array(extra_negative_points).reshape(-1, 3)
        points = np.concatenate((points, extra_negative_points), axis=0)
        labels = np.concatenate((labels, np.zeros((extra_negative_points.shape[0]))), axis=0)
        # print('extra_negative_points ', extra_negative_points, extra_negative_points.shape)
        # print('==> points ', points.shape, labels)
    
    if fix_extra_point_num is None:
        left_point_num = num_positive_extra + num_negative_extra + 1 - labels.shape[0]
    else:
        left_point_num = fix_extra_point_num  + 1 - labels.shape[0]
    
    for _ in range(left_point_num):
        ignore_point = np.array([-1,-1,-1]).reshape(1, 3)
        points = np.concatenate((points, ignore_point), axis=0)
        labels = np.concatenate((labels, np.array([-1]).reshape(1)), axis=0)
    
    return (points, labels)


def predict_cases(model, list_of_lists, seg_of_lists, output_filenames, folds, save_npz, num_threads_preprocessing,
                  num_threads_nifti_save, segs_from_prev_stage=None, do_tta=True, mixed_precision=True,
                  overwrite_existing=False,
                  all_in_gpu=False, step_size=0.5, checkpoint_name="model_final_checkpoint",
                  segmentation_export_kwargs: dict = None, disable_postprocessing: bool = False,
                  use_point_prompt=False, use_box_prompt=False):
    """
    :param segmentation_export_kwargs:
    :param model: folder where the model is saved, must contain fold_x subfolders
    :param list_of_lists: [[case0_0000.nii.gz, case0_0001.nii.gz], [case1_0000.nii.gz, case1_0001.nii.gz], ...]
    :param output_filenames: [output_file_case0.nii.gz, output_file_case1.nii.gz, ...]
    :param seg_of_lists: [[seg_case0.nii.gz], [seg_case1.nii.gz], ...]
    :param folds: default: (0, 1, 2, 3, 4) (but can also be 'all' or a subset of the five folds, for example use (0, )
    for using only fold_0
    :param save_npz: default: False
    :param num_threads_preprocessing:
    :param num_threads_nifti_save:
    :param segs_from_prev_stage:
    :param do_tta: default: True, can be set to False for a 8x speedup at the cost of a reduced segmentation quality
    :param overwrite_existing: default: True
    :param mixed_precision: if None then we take no action. If True/False we overwrite what the model has in its init
    :return:
    """
    assert len(list_of_lists) == len(output_filenames)
    if segs_from_prev_stage is not None: assert len(segs_from_prev_stage) == len(output_filenames)

    pool = Pool(num_threads_nifti_save)
    results = []

    cleaned_output_files = []
    for o in output_filenames:
        dr, f = os.path.split(o)
        if len(dr) > 0:
            maybe_mkdir_p(dr)
        if not f.endswith(".nii.gz"):
            f, _ = os.path.splitext(f)
            f = f + ".nii.gz"
        cleaned_output_files.append(join(dr, f))

    if not overwrite_existing:
        print("number of cases:", len(list_of_lists))
        # if save_npz=True then we should also check for missing npz files
        not_done_idx = [i for i, j in enumerate(cleaned_output_files) if (not isfile(j)) or (save_npz and not isfile(j[:-7] + '.npz'))]

        cleaned_output_files = [cleaned_output_files[i] for i in not_done_idx]
        list_of_lists = [list_of_lists[i] for i in not_done_idx]
        if segs_from_prev_stage is not None:
            segs_from_prev_stage = [segs_from_prev_stage[i] for i in not_done_idx]

        print("number of cases that still need to be predicted:", len(cleaned_output_files))

    print("emptying cuda cache")
    torch.cuda.empty_cache()

    print("loading parameters for folds,", folds)
    trainer, params = load_model_and_checkpoint_files(model, folds, mixed_precision=mixed_precision,
                                                      checkpoint_name=checkpoint_name)
    # trainer.load_latest_checkpoint(False)
    trainer.load_checkpoint_ram(params[0], False)
    if segmentation_export_kwargs is None:
        if 'segmentation_export_params' in trainer.plans.keys():
            force_separate_z = trainer.plans['segmentation_export_params']['force_separate_z']
            interpolation_order = trainer.plans['segmentation_export_params']['interpolation_order']
            interpolation_order_z = trainer.plans['segmentation_export_params']['interpolation_order_z']
        else:
            force_separate_z = None
            interpolation_order = 1
            interpolation_order_z = 0
    else:
        force_separate_z = segmentation_export_kwargs['force_separate_z']
        interpolation_order = segmentation_export_kwargs['interpolation_order']
        interpolation_order_z = segmentation_export_kwargs['interpolation_order_z']

    print("starting preprocessing generator")
    preprocessing = preprocess_multithreaded(trainer, list_of_lists, seg_of_lists, cleaned_output_files, num_threads_preprocessing,
                                             segs_from_prev_stage)
    print("starting prediction...")

    all_output_files = []
    for case_files, seg_file, output_filename in zip(list_of_lists, seg_of_lists, output_filenames):
        data, seg, dct = trainer.preprocess_patient(case_files, seg_file)
        # for preprocessed in preprocessing:
        #     output_filename, (data, seg, dct) = preprocessed
        all_output_files.append(all_output_files)

        print("predicting", output_filename.split("/")[-1])
        print(f"build box {use_box_prompt} and points {use_point_prompt} prompts for", output_filename.split("/")[-1])
    
        ### resizes the data to the model's input size
        patch_size = trainer.patch_size
    
        original_shape = data.shape
        #### crop the data to the patch size according to the bbox of the segmentation
        ones_idx = np.where(seg[0] > 0.5)
        min_pos = [np.min(ones_idx[i]) for i in range(3)]
        max_pos = [np.max(ones_idx[i]) for i in range(3)]

        ### select the bbox of the segmentation
        ### crop the data such that the bbox is included in every patch
        crop_bbox = []
        if use_box_prompt or use_point_prompt:
            for i in range(3):
                ### crop the data to the bbox such that the bbox is included in every patch
                center_patch = (min_pos[i] + max_pos[i]) // 2
                max_boundary = min(data[0].shape[i]-center_patch, center_patch)
                left_patch = center_patch - max_boundary
                # left_patch = min_pos[i] - patch_size[i] // 2
                right_patch = center_patch + max_boundary
                # right_patch = max_pos[i] + patch_size[i] // 2
                crop_bbox.append([left_patch, right_patch])
        else:
            ### If we do have box infor, then do not use it
            crop_bbox = [[0, seg[0].shape[i]] for i in range(3)]

        print("crop_bbox: ", crop_bbox)
        x_min, x_max, y_min, y_max, z_min, z_max = crop_bbox[0][0], crop_bbox[0][1], crop_bbox[1][0], crop_bbox[1][1], crop_bbox[2][0], crop_bbox[2][1]
        data_crop = data[:, x_min:x_max, y_min:y_max, z_min:z_max]
        seg_crop = seg[:, x_min:x_max, y_min:y_max, z_min:z_max]

        # # for sliding window inference the image must at least be as large as the patch size. It does not matter
        # # whether the shape is divisible by 2**num_pool as long as the patch size is
        pad_border_mode = 'constant'
        pad_kwargs = {'constant_values': 0}
        
        data_resized, slicer = pad_nd_image(data_crop, patch_size, pad_border_mode, pad_kwargs, True, None)
        seg_resized, _ = pad_nd_image(seg_crop, patch_size, pad_border_mode, pad_kwargs, True, None)
        pad_shape = data_resized[0].shape
        data_resized, seg_resized = resize_multichannel_image(data_resized, patch_size), \
                                    np.expand_dims(resize_segmentation(seg_resized[0], patch_size, 0), 0)
        seg_resized = (seg_resized > 0.5).astype(np.int64)

        #### 
        points_single = None
        box_single = None
        points_num = 10
        if use_point_prompt:
            point_np, point_label_np = select_points_np(seg_resized[0], num_positive_extra=points_num, num_negative_extra=points_num)
            point_np, point_label_np = np.expand_dims(point_np, 0), np.expand_dims(point_label_np, 0)
            points_single = (point_np, point_label_np)

        if use_box_prompt:
            box_single_np = generate_box_np(seg_resized[0])
            box_single_np = np.expand_dims(box_single_np, 0)
            box_single = box_single_np
        
        # points_single = None
        # softmax = trainer.predict_with_prompt(data_resized.unsqueeze(0))
        softmax = trainer.predict_preprocessed_data_return_seg_and_softmax_withbox_points(
            data_resized, box_single, points_single, do_mirroring=do_tta, mirror_axes=trainer.data_aug_params['mirror_axes'], use_sliding_window=True,
            step_size=step_size, use_gaussian=True, all_in_gpu=all_in_gpu,
            mixed_precision=mixed_precision)[1]
        # import pdb; pdb.set_trace()
        ### resize the softmax to the original shape
        softmax = resize_multichannel_image(softmax, pad_shape)
        
        slicer = tuple(
            [slice(0, softmax.shape[i]) for i in
             range(len(softmax.shape) - (len(slicer) - 1))] + slicer[1:])
        softmax = softmax[slicer]

        seg_before_crop = np.concatenate([np.ones_like(seg), *[np.zeros_like(seg) for _ in range(softmax.shape[0]-1)]], 0)
        seg_before_crop[:, x_min:x_max, y_min:y_max, z_min:z_max] = softmax

        # softmax = resize_multichannel_image(softmax, 
        #                                     original_shape)
        ### save the maximum slice of the segmentation in png
        # import pdb; pdb.set_trace()
        softmax = seg_before_crop
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        # seg_before_crop = np.concatenate([np.ones_like(seg), *[np.zeros_like(seg) for _ in range(softmax.shape[0]-1)]], 0)
        # seg_before_crop[:, x_min:x_max, y_min:y_max, z_min:z_max] = softmax
    
        max_slice = np.argmax(np.sum(softmax[1], axis=(1, 2)))
        plt.figure()
        plt.imshow(softmax[1, max_slice], cmap='gray')
        ### show the bbox of the segmentation
        # bboxes = box_single[0]
        # x_min, x_max, y_min, y_max, z_min, z_max = bboxes[0], bboxes[3], bboxes[1], bboxes[4], bboxes[2], bboxes[5]
        # ## the y, z axis is reversed in the plot
        # rect = patches.Rectangle((z_min, y_min), z_max-z_min, y_max-y_min, linewidth=1, edgecolor='r', facecolor='none')
        # plt.gca().add_patch(rect)
        plt.savefig(os.path.join("/home/zze3980/projects/medpretrain/nnUNetDG/temp_logs",
                                 output_filename.split("/")[-1][:-7] + "_max_slice.png"))

        # softmax = seg_before_crop
        # print("Volume after prediction: ", np.sum(softmax[1]>0.5))
        transpose_forward = trainer.plans.get('transpose_forward')
        if transpose_forward is not None:
            transpose_backward = trainer.plans.get('transpose_backward')
            softmax = softmax.transpose([0] + [i + 1 for i in transpose_backward])

        if save_npz:
            npz_file = output_filename[:-7] + ".npz"
        else:
            npz_file = None

        if hasattr(trainer, 'regions_class_order'):
            region_class_order = trainer.regions_class_order
        else:
            region_class_order = None

        """There is a problem with python process communication that prevents us from communicating objects 
        larger than 2 GB between processes (basically when the length of the pickle string that will be sent is 
        communicated by the multiprocessing.Pipe object then the placeholder (I think) does not allow for long 
        enough strings (lol). This could be fixed by changing i to l (for long) but that would require manually 
        patching system python code. We circumvent that problem here by saving softmax_pred to a npy file that will 
        then be read (and finally deleted) by the Process. save_segmentation_nifti_from_softmax can take either 
        filename or np.ndarray and will handle this automatically"""
        bytes_per_voxel = 4
        if all_in_gpu:
            bytes_per_voxel = 2  # if all_in_gpu then the return value is half (float16)
        if np.prod(softmax.shape) > (2e9 / bytes_per_voxel * 0.85):  # * 0.85 just to be save
            print(
                "This output is too large for python process-process communication. Saving output temporarily to disk")
            np.save(output_filename[:-7] + ".npy", softmax)
            softmax = output_filename[:-7] + ".npy"
        # save_segmentation_nifti_from_softmax(softmax, output_filename, dct, interpolation_order, region_class_order,
        #                                     None, None,
        #                                     npz_file, None, force_separate_z, interpolation_order_z)

        results.append(pool.starmap_async(save_segmentation_nifti_from_softmax,
                                          ((softmax, output_filename, dct, interpolation_order, region_class_order,
                                            None, None,
                                            npz_file, None, force_separate_z, interpolation_order_z),)
                                          ))

    print("inference done. Now waiting for the segmentation export to finish...")

    _ = [i.get() for i in results]
    # now apply postprocessing
    # first load the postprocessing properties if they are present. Else raise a well visible warning
    if not disable_postprocessing:
        results = []
        pp_file = join(model, "postprocessing.json")
        if isfile(pp_file):
            print("postprocessing...")
            shutil.copy(pp_file, os.path.abspath(os.path.dirname(output_filenames[0])))
            # for_which_classes stores for which of the classes everything but the largest connected component needs to be
            # removed
            for_which_classes, min_valid_obj_size = load_postprocessing(pp_file)
            results.append(pool.starmap_async(load_remove_save,
                                              zip(output_filenames, output_filenames,
                                                  [for_which_classes] * len(output_filenames),
                                                  [min_valid_obj_size] * len(output_filenames))))
            _ = [i.get() for i in results]
        else:
            print("WARNING! Cannot run postprocessing because the postprocessing file is missing. Make sure to run "
                  "consolidate_folds in the output folder of the model first!\nThe folder you need to run this in is "
                  "%s" % model)

    pool.close()
    pool.join()


def check_input_folder_and_return_caseIDs(input_folder, expected_num_modalities):
    print("This model expects %d input modalities for each image" % expected_num_modalities)
    files = subfiles(input_folder, suffix=".nii.gz", join=False, sort=True)

    maybe_case_ids = np.unique([i[:-12] for i in files])

    remaining = deepcopy(files)
    missing = []

    assert len(files) > 0, "input folder did not contain any images (expected to find .nii.gz file endings)"

    # now check if all required files are present and that no unexpected files are remaining
    for c in maybe_case_ids:
        for n in range(expected_num_modalities):
            expected_output_file = c + "_%04.0d.nii.gz" % n
            if not isfile(join(input_folder, expected_output_file)):
                missing.append(expected_output_file)
            else:
                remaining.remove(expected_output_file)

    print("Found %d unique case ids, here are some examples:" % len(maybe_case_ids),
          np.random.choice(maybe_case_ids, min(len(maybe_case_ids), 10)))
    print("If they don't look right, make sure to double check your filenames. They must end with _0000.nii.gz etc")

    if len(remaining) > 0:
        print("found %d unexpected remaining files in the folder. Here are some examples:" % len(remaining),
              np.random.choice(remaining, min(len(remaining), 10)))

    if len(missing) > 0:
        print("Some files are missing:")
        print(missing)
        raise RuntimeError("missing files in input_folder")

    return maybe_case_ids


def predict_from_folder(model: str, input_folder: str, label_folder: str, output_folder: str, folds: Union[Tuple[int], List[int]],
                                    save_npz: bool, num_threads_preprocessing: int, num_threads_nifti_save: int,
                                    lowres_segmentations: Union[str, None],
                                    part_id: int, num_parts: int, tta: bool, mixed_precision: bool = True,
                                    overwrite_existing: bool = True, mode: str = 'normal', overwrite_all_in_gpu: bool = None,
                                    step_size: float = 0.5, checkpoint_name: str = "model_final_checkpoint",
                                    segmentation_export_kwargs: dict = None, disable_postprocessing: bool = False,
                                    use_point_prompt: bool = False, use_box_prompt: bool = False):
    """
        here we use the standard naming scheme to generate list_of_lists and output_files needed by predict_cases

    :param model:
    :param input_folder:
    :param output_folder:
    :param folds:
    :param save_npz:
    :param num_threads_preprocessing:
    :param num_threads_nifti_save:
    :param lowres_segmentations:
    :param part_id:
    :param num_parts:
    :param tta:
    :param mixed_precision:
    :param overwrite_existing: if not None then it will be overwritten with whatever is in there. None is default (no overwrite)
    :return:
    """
    maybe_mkdir_p(output_folder)
    shutil.copy(join(model, 'plans.pkl'), output_folder)

    assert isfile(join(model, "plans.pkl")), "Folder with saved model weights must contain a plans.pkl file"
    expected_num_modalities = load_pickle(join(model, "plans.pkl"))['num_modalities']

    # check input folder integrity
    case_ids = check_input_folder_and_return_caseIDs(input_folder, expected_num_modalities)

    output_files = [join(output_folder, i + ".nii.gz") for i in case_ids]
    label_files = [join(label_folder, i + ".nii.gz") for i in case_ids]

    all_files = subfiles(input_folder, suffix=".nii.gz", join=False, sort=True)
    list_of_lists = [[join(input_folder, i) for i in all_files if i[:len(j)].startswith(j) and
                      len(i) == (len(j) + 12)] for j in case_ids]

    if lowres_segmentations is not None:
        assert isdir(lowres_segmentations), "if lowres_segmentations is not None then it must point to a directory"
        lowres_segmentations = [join(lowres_segmentations, i + ".nii.gz") for i in case_ids]
        assert all([isfile(i) for i in lowres_segmentations]), "not all lowres_segmentations files are present. " \
                                                               "(I was searching for case_id.nii.gz in that folder)"
        lowres_segmentations = lowres_segmentations[part_id::num_parts]
    else:
        lowres_segmentations = None

    if mode == "normal":
        if overwrite_all_in_gpu is None:
            all_in_gpu = False
        else:
            all_in_gpu = overwrite_all_in_gpu

        return predict_cases(model, list_of_lists[part_id::num_parts], label_files[part_id::num_parts], output_files[part_id::num_parts], folds,
                             save_npz, num_threads_preprocessing, num_threads_nifti_save, lowres_segmentations, tta,
                             mixed_precision=mixed_precision, overwrite_existing=overwrite_existing,
                             all_in_gpu=all_in_gpu,
                             step_size=step_size, checkpoint_name=checkpoint_name,
                             segmentation_export_kwargs=segmentation_export_kwargs,
                             disable_postprocessing=disable_postprocessing,
                             use_point_prompt=use_point_prompt, use_box_prompt=use_box_prompt)
    else:
        raise ValueError("unrecognized mode. Must be normal, fast or fastest")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", '--input_folder', help="Must contain all modalities for each patient in the correct"
                                                     " order (same as training). Files must be named "
                                                     "CASENAME_XXXX.nii.gz where XXXX is the modality "
                                                     "identifier (0000, 0001, etc)", required=True)
    parser.add_argument('-l', "--label_folder", required=True, help="folder for saving the annotations")

    parser.add_argument('-o', "--output_folder", required=True, help="folder for saving predictions")
    parser.add_argument('-t', '--task_name', help='task name or task ID, required.',
                        default=default_plans_identifier, required=True)

    parser.add_argument('-tr', '--trainer_class_name',
                        help='Name of the nnUNetTrainer used for 2D U-Net, full resolution 3D U-Net and low resolution '
                             'U-Net. The default is %s. You need to specify this argument if using any trainer other than %s. If you are running inference with the cascade and the folder '
                             'pointed to by --lowres_segmentations does not contain the segmentation maps generated by '
                             'the low resolution U-Net then the low resolution segmentation maps will be automatically '
                             'generated. For this case, make sure to set the trainer class here that matches your '
                             '--cascade_trainer_class_name (this part can be ignored if defaults are used).'
                             % (default_trainer,default_trainer),
                        required=False,
                        default=default_trainer)
    parser.add_argument('-ctr', '--cascade_trainer_class_name',
                        help="Trainer class name used for predicting the 3D full resolution U-Net part of the cascade."
                             "Default is %s" % default_cascade_trainer, required=False,
                        default=default_cascade_trainer)

    parser.add_argument('-m', '--model', help="2d, 3d_lowres, 3d_fullres or 3d_cascade_fullres. Default: 3d_fullres",
                        default="3d_fullres", required=False)

    parser.add_argument('-p', '--plans_identifier', help='do not touch this unless you know what you are doing',
                        default=default_plans_identifier, required=False)

    parser.add_argument('-f', '--folds', nargs='+', default='None',
                        help="folds to use for prediction. Default is None which means that folds will be detected "
                             "automatically in the model output folder")

    parser.add_argument('-z', '--save_npz', required=False, action='store_true',
                        help="use this if you want to ensemble these predictions with those of other models. Softmax "
                             "probabilities will be saved as compressed numpy arrays in output_folder and can be "
                             "merged between output_folders with nnUNet_ensemble_predictions")

    parser.add_argument('--lowres_segmentations', required=False, default='None',
                        help="if model is the highres stage of the cascade then you can use this folder to provide "
                             "predictions from the low resolution 3D U-Net. If this is left at default, the "
                             "predictions will be generated automatically (provided that the 3D low resolution U-Net "
                             "network weights are present")

    parser.add_argument("--part_id", type=int, required=False, default=0, help="Used to parallelize the prediction of "
                                                                               "the folder over several GPUs. If you "
                                                                               "want to use n GPUs to predict this "
                                                                               "folder you need to run this command "
                                                                               "n times with --part_id=0, ... n-1 and "
                                                                               "--num_parts=n (each with a different "
                                                                               "GPU (for example via "
                                                                               "CUDA_VISIBLE_DEVICES=X)")

    parser.add_argument("--num_parts", type=int, required=False, default=1,
                        help="Used to parallelize the prediction of "
                             "the folder over several GPUs. If you "
                             "want to use n GPUs to predict this "
                             "folder you need to run this command "
                             "n times with --part_id=0, ... n-1 and "
                             "--num_parts=n (each with a different "
                             "GPU (via "
                             "CUDA_VISIBLE_DEVICES=X)")

    parser.add_argument("--num_threads_preprocessing", required=False, default=6, type=int, help=
    "Determines many background processes will be used for data preprocessing. Reduce this if you "
    "run into out of memory (RAM) problems. Default: 6")

    parser.add_argument("--num_threads_nifti_save", required=False, default=2, type=int, help=
    "Determines many background processes will be used for segmentation export. Reduce this if you "
    "run into out of memory (RAM) problems. Default: 2")

    parser.add_argument("--disable_tta", required=False, default=False, action="store_true",
                        help="set this flag to disable test time data augmentation via mirroring. Speeds up inference "
                             "by roughly factor 4 (2D) or 8 (3D)")

    parser.add_argument("--overwrite_existing", required=False, default=False, action="store_true",
                        help="Set this flag if the target folder contains predictions that you would like to overwrite")

    parser.add_argument("--mode", type=str, default="normal", required=False, help="Hands off!")
    parser.add_argument("--all_in_gpu", type=str, default="None", required=False, help="can be None, False or True. "
                                                                                       "Do not touch.")
    parser.add_argument("--step_size", type=float, default=0.5, required=False, help="don't touch")
    # parser.add_argument("--interp_order", required=False, default=3, type=int,
    #                     help="order of interpolation for segmentations, has no effect if mode=fastest. Do not touch this.")
    # parser.add_argument("--interp_order_z", required=False, default=0, type=int,
    #                     help="order of interpolation along z is z is done differently. Do not touch this.")
    # parser.add_argument("--force_separate_z", required=False, default="None", type=str,
    #                     help="force_separate_z resampling. Can be None, True or False, has no effect if mode=fastest. "
    #                          "Do not touch this.")
    parser.add_argument('-chk',
                        help='checkpoint name, default: model_final_checkpoint',
                        required=False,
                        default='model_final_checkpoint')
    parser.add_argument('--disable_mixed_precision', default=False, action='store_true', required=False,
                        help='Predictions are done with mixed precision by default. This improves speed and reduces '
                             'the required vram. If you want to disable mixed precision you can set this flag. Note '
                             'that this is not recommended (mixed precision is ~2x faster!)')
    parser.add_argument("--use_point_prompt", required=False, default=False, action="store_true",
                        help="use point prompt")
    parser.add_argument("--use_box_prompt", required=False, default=False, action="store_true",
                        help="use box prompt")

    args = parser.parse_args()
    input_folder = args.input_folder
    label_folder = args.label_folder
    output_folder = args.output_folder
    part_id = args.part_id
    num_parts = args.num_parts
    folds = args.folds
    save_npz = args.save_npz
    lowres_segmentations = args.lowres_segmentations
    num_threads_preprocessing = args.num_threads_preprocessing
    num_threads_nifti_save = args.num_threads_nifti_save
    disable_tta = args.disable_tta
    step_size = args.step_size
    # interp_order = args.interp_order
    # interp_order_z = args.interp_order_z
    # force_separate_z = args.force_separate_z
    overwrite_existing = args.overwrite_existing
    mode = args.mode
    all_in_gpu = args.all_in_gpu
    model = args.model
    trainer_class_name = args.trainer_class_name
    cascade_trainer_class_name = args.cascade_trainer_class_name

    task_name = args.task_name
    use_point_prompt = args.use_point_prompt
    use_box_prompt = args.use_box_prompt

    if not task_name.startswith("Task"):
        task_id = int(task_name)
        task_name = convert_id_to_task_name(task_id)

    assert model in ["2d", "3d_lowres", "3d_fullres", "3d_cascade_fullres"], "-m must be 2d, 3d_lowres, 3d_fullres or " \
                                                                             "3d_cascade_fullres"

    # if force_separate_z == "None":
    #     force_separate_z = None
    # elif force_separate_z == "False":
    #     force_separate_z = False
    # elif force_separate_z == "True":
    #     force_separate_z = True
    # else:
    #     raise ValueError("force_separate_z must be None, True or False. Given: %s" % force_separate_z)

    if lowres_segmentations == "None":
        lowres_segmentations = None

    if isinstance(folds, list):
        if folds[0] == 'all' and len(folds) == 1:
            pass
        else:
            folds = [int(i) for i in folds]
    elif folds == "None":
        folds = None
    else:
        raise ValueError("Unexpected value for argument folds")

    assert all_in_gpu in ['None', 'False', 'True']
    if all_in_gpu == "None":
        all_in_gpu = None
    elif all_in_gpu == "True":
        all_in_gpu = True
    elif all_in_gpu == "False":
        all_in_gpu = False

    if model == "3d_cascade_fullres":
        trainer = cascade_trainer_class_name
    else:
        trainer = trainer_class_name

    model_folder_name = join(network_training_output_dir, model, task_name, trainer + "__" +
                              args.plans_identifier)
    print("using model stored in ", model_folder_name)
    assert isdir(model_folder_name), "model output folder not found. Expected: %s" % model_folder_name

    st = time()
    print("Predicting using nnUNet with", use_box_prompt, use_point_prompt)
    predict_from_folder(model_folder_name, input_folder, label_folder, output_folder, folds, save_npz, num_threads_preprocessing,
                        num_threads_nifti_save, lowres_segmentations, part_id, num_parts, not disable_tta,
                        overwrite_existing=overwrite_existing, mode=mode, overwrite_all_in_gpu=all_in_gpu,
                        mixed_precision=not args.disable_mixed_precision,
                        step_size=step_size, checkpoint_name=args.chk,
                        use_box_prompt=use_box_prompt, use_point_prompt=use_point_prompt)
    end = time()
    save_json(end - st, join(output_folder, 'prediction_time.txt'))

if __name__ == "__main__":
    main()
