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


from collections import OrderedDict
from typing import Tuple

import numpy as np
import torch
import copy
import random
import nnunet
from nnunet.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from nnunet.network_architecture.generic_PromptUNet import Generic_PromptUNet
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.training.data_augmentation.default_data_augmentation import default_2D_augmentation_params, \
    get_patch_size, default_3D_augmentation_params
from nnunet.training.dataloading.dataset_loading import unpack_dataset, DataLoader3D, DataLoader2D
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.utilities.nd_softmax import softmax_helper
from sklearn.model_selection import KFold
from torch import nn
from torch.cuda.amp import autocast
from nnunet.training.learning_rate.poly_lr import poly_lr
from batchgenerators.utilities.file_and_folder_operations import *


class nnUNetPromptTrainerV2(nnUNetTrainer):
    """
    Info for Fabian: same as internal nnUNetTrainerV2_2
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = 200
        # self.initial_lr = 1e-2
        self.initial_lr = 1e-2
        self.deep_supervision_scales = None
        self.ds_loss_weights = None

        self.pin_memory = True
        ### save every 100 epochs
        self.save_every = 100
        self.save_latest_only = False  # if false it will not store/overwrite _latest but separate files each
        # time an intermediate checkpoint is created
        self.save_intermediate_checkpoints = True
    
    def initialize(self, training=True, force_load_plans=False):
        """
        - replaced get_default_augmentation with get_moreDA_augmentation
        - enforce to only run this code once
        - loss function wrapper for deep supervision

        :param training:
        :param force_load_plans:
        :return:
        """
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()

            self.process_plans(self.plans)

            self.setup_DA_params()

            ################# Here we wrap the loss for deep supervision ############
            # we need to know the number of outputs of the network
            net_numpool = len(self.net_num_pool_op_kernel_sizes)

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
            weights[~mask] = 0
            weights = weights / weights.sum()
            self.ds_loss_weights = weights
            # now wrap the loss
            self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)
            ################# END ###################

            self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +
                                                      "_stage%d" % self.stage)
            if training:
                self.dl_tr, self.dl_val = self.get_basic_generators()
                if self.unpack_data:
                    print("unpacking dataset")
                    unpack_dataset(self.folder_with_preprocessed_data)
                    print("done")
                else:
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!")

                self.tr_gen, self.val_gen = get_moreDA_augmentation(
                    self.dl_tr, self.dl_val,
                    self.data_aug_params[
                        'patch_size_for_spatialtransform'],
                    self.data_aug_params,
                    deep_supervision_scales=self.deep_supervision_scales,
                    pin_memory=self.pin_memory,
                    use_nondetMultiThreadedAugmenter=False
                )
                self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                                       also_print_to_console=False)
                self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                                       also_print_to_console=False)
            else:
                pass

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()

            assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True

    def initialize_network(self):
        """
        - momentum 0.99
        - SGD instead of Adam
        - self.lr_scheduler = None because we do poly_lr
        - deep supervision = True
        - i am sure I forgot something here

        Known issue: forgot to set neg_slope=0 in InitWeights_He; should not make a difference though
        :return:
        """
        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d

        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.InstanceNorm2d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        text_embedding = None
        if "text_embedding" in self.plans.keys():
            text_embedding_dict = pickle.load(open(self.plans["text_embedding"], 'rb'))
            organ_name = self.plans["organ_name"]
            print("Using text embedding for organ", organ_name)
            text_embedding = text_embedding_dict[organ_name]

        self.network = Generic_PromptUNet(self.num_input_channels, self.base_num_features, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                    dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True,
                                    transformer_dim=768,embedding_nums=3,text_embedding=text_embedding)
        if "pretrained" in self.plans.keys():
            ### load the pretrained model
            print("Loading pretrained model from", self.plans["pretrained"])
            self.network.load_state_dict(torch.load(self.plans["pretrained"], weights_only=True,
                                                    map_location=torch.device('cpu')), strict=False)

        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        ### only finetune the last several localization blocks
        # params = list(self.network.seg_outputs.parameters()) + list(self.network.conv_blocks_localization[2:].parameters())
        # params = self.network.parameters()
        ### for prompts-tuning
        # params = [self.network.mask_tokens.weight,
        #           self.network.text_embedding] + list(self.network.seg_outputs.parameters())

        ### for prompts-decoder
        # params = [self.network.mask_tokens.weight,
        #          self.network.text_embedding] + list(self.network.seg_outputs.parameters()) + list(self.network.tu[2:].parameters())
        
        ### load the pretrained model
        bottom_level = self.plans["finetune_block"]
        params = [self.network.mask_tokens.weight, self.network.text_embedding] + list(self.network.seg_outputs.parameters()) + \
            list(self.network.tu[2:].parameters()) + list(self.network.conv_blocks_localization[:bottom_level].parameters())
        
        # self.optimizer = torch.optim.SGD(params, self.initial_lr, weight_decay=self.weight_decay,
        #                                  momentum=0.99, nesterov=True)
        self.optimizer = torch.optim.AdamW(params, self.initial_lr,
                                           weight_decay=self.weight_decay)
        self.lr_scheduler = None

    def preprocess_patient(self, input_files, seg_file=None,):
       
        from nnunet.training.model_restore import recursive_find_python_class
        preprocessor_name = self.plans.get('preprocessor_name')
        if preprocessor_name is None:
            if self.threeD:
                preprocessor_name = "GenericPreprocessor"
            else:
                preprocessor_name = "PreprocessorFor2D"

        print("using preprocessor", preprocessor_name)
        preprocessor_class = recursive_find_python_class([join(nnunet.__path__[0], "preprocessing")],
                                                         preprocessor_name,
                                                         current_module="nnunet.preprocessing")
        assert preprocessor_class is not None, "Could not find preprocessor %s in nnunet.preprocessing" % \
                                               preprocessor_name
        preprocessor = preprocessor_class(self.normalization_schemes, self.use_mask_for_norm,
                                          self.transpose_forward, self.intensity_properties)

        d, s, properties = preprocessor.preprocess_test_case(input_files,
                                                             self.plans['plans_per_stage'][self.stage][
                                                                 'current_spacing'], seg_file)
        return d, s, properties

    def run_online_evaluation(self, output, target):
        """
        due to deep supervision the return value and the reference are now lists of tensors. We only need the full
        resolution output because this is what we are interested in in the end. The others are ignored
        :param output:
        :param target:
        :return:
        """
        target = target[0]
        output = output[0]
        return super().run_online_evaluation(output, target)

    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True,
                 step_size: float = 0.5, save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True):
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """
        ds = self.network.do_ds
        self.network.do_ds = False
        ret = super().validate(do_mirroring=do_mirroring, use_sliding_window=use_sliding_window, step_size=step_size,
                               save_softmax=save_softmax, use_gaussian=use_gaussian,
                               overwrite=overwrite, validation_folder_name=validation_folder_name, debug=debug,
                               all_in_gpu=all_in_gpu, segmentation_export_kwargs=segmentation_export_kwargs,
                               run_postprocessing_on_folds=run_postprocessing_on_folds)

        self.network.do_ds = ds
        return ret
    
    def validate_test(self, do_mirroring: bool = True, use_sliding_window: bool = True,
                 step_size: float = 0.5, save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'test_raw', debug: bool = False, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True):
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """
        ds = self.network.do_ds
        self.network.do_ds = False
        val_data = copy.deepcopy(self.dataset_val)
        test_data = copy.deepcopy(self.dataset_test)

        self.dataset_val = test_data
        ret = super().validate(do_mirroring=do_mirroring, use_sliding_window=use_sliding_window, step_size=step_size,
                               save_softmax=save_softmax, use_gaussian=use_gaussian,
                               overwrite=overwrite, validation_folder_name=validation_folder_name, debug=debug,
                               all_in_gpu=all_in_gpu, segmentation_export_kwargs=segmentation_export_kwargs,
                               run_postprocessing_on_folds=run_postprocessing_on_folds)
        self.network.do_ds = ds

        ### restore val data
        self.dataset_val = val_data

        return ret

    def predict_preprocessed_data_return_seg_and_softmax(self, data: np.ndarray, do_mirroring: bool = True,
                                                         mirror_axes: Tuple[int] = None,
                                                         use_sliding_window: bool = True, step_size: float = 0.5,
                                                         use_gaussian: bool = True, pad_border_mode: str = 'constant',
                                                         pad_kwargs: dict = None, all_in_gpu: bool = False,
                                                         verbose: bool = True, mixed_precision=True) -> Tuple[np.ndarray, np.ndarray]:
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """
        ds = self.network.do_ds
        self.network.do_ds = False
        ret = super().predict_preprocessed_data_return_seg_and_softmax(data,
                                                                       do_mirroring=do_mirroring,
                                                                       mirror_axes=mirror_axes,
                                                                       use_sliding_window=use_sliding_window,
                                                                       step_size=step_size, use_gaussian=use_gaussian,
                                                                       pad_border_mode=pad_border_mode,
                                                                       pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu,
                                                                       verbose=verbose,
                                                                       mixed_precision=mixed_precision)
        self.network.do_ds = ds
        return ret

    def do_split(self):
        """
        The default split is a 5 fold CV on all available training cases. nnU-Net will create a split (it is seeded,
        so always the same) and save it as splits_final.pkl file in the preprocessed data directory.
        Sometimes you may want to create your own split for various reasons. For this you will need to create your own
        splits_final.pkl file. If this file is present, nnU-Net is going to use it and whatever splits are defined in
        it. You can create as many splits in this file as you want. Note that if you define only 4 splits (fold 0-3)
        and then set fold=4 when training (that would be the fifth split), nnU-Net will print a warning and proceed to
        use a random 80:20 data split.
        :return:
        """
        if self.fold == "all":
            # if fold==all then we use all images for training and validation
            tr_keys = val_keys = list(self.dataset.keys())
        elif "domain" in str(self.fold):
            splits_file = join(self.dataset_directory, "splits_domain.pkl")

            # if the split file does not exist we need to create it
            if not isfile(splits_file):
                self.print_to_log_file("Creating new standard cross domain with train val test split...")
                splits = []
                all_keys_sorted = np.sort(list(self.dataset.keys()))
                all_domain_list = list(set([i.split("_")[0] for i in all_keys_sorted]))
                all_domain_list.sort()
                total_domain_num = len(all_domain_list)

                for i in range(total_domain_num):
                    test_domain = all_domain_list[i]
                    if i == 0:
                        train_domain = all_domain_list[1:]
                    elif i == total_domain_num - 1:
                        train_domain = all_domain_list[:-1]
                    else:
                        train_domain = all_domain_list[:i] + all_domain_list[i+1:]
                    train_keys_combined = [j for j in all_keys_sorted if j.split("_")[0] in train_domain]
                    test_keys = [j for j in all_keys_sorted if j.split("_")[0] == test_domain]
                    test_keys = np.array(test_keys)

                    ### split the train_keys into train and val
                    kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
                    train_idx, val_idx = kfold.split(train_keys_combined).__next__()
                    train_keys = np.array(train_keys_combined)[train_idx]
                    val_keys = np.array(train_keys_combined)[val_idx]

                    splits.append(OrderedDict())
                    splits[-1]['train'] = train_keys
                    splits[-1]['val'] = val_keys
                    splits[-1]['test'] = test_keys
    
                save_pickle(splits, splits_file)

            else:
                self.print_to_log_file("Using splits from existing split file:", splits_file)
                splits = load_pickle(splits_file)
                self.print_to_log_file("The split file contains %d splits." % len(splits))

            fold_id = int(self.fold.replace("domain", ""))
            self.print_to_log_file("Desired fold for training: %d" % fold_id)
            if fold_id < len(splits):
                tr_keys = splits[fold_id]['train']
                val_keys = splits[fold_id]['val']
                test_keys = splits[fold_id]['test']
                self.print_to_log_file("This split has %d training, %d validation, and %d test cases."
                                       % (len(tr_keys), len(val_keys), len(test_keys))
                                       )
            else:
                assert fold_id < len(splits), "You requested fold %d but splits file has %d splits" % (fold_id, len(splits))
    
        else:
            splits_file = join(self.dataset_directory, "splits_final.pkl")

            # if the split file does not exist we need to create it
            if not isfile(splits_file):
                self.print_to_log_file("Creating new 5-fold cross-validation with train val test split...")
                splits = []
                all_keys_sorted = np.sort(list(self.dataset.keys()))
                kfold = KFold(n_splits=6, shuffle=True, random_state=12345)
                for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
                    train_keys_combined = np.array(all_keys_sorted)[train_idx]
                    test_keys = np.array(all_keys_sorted)[test_idx]

                    ### further split train_keys into train and val
                    kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
                    train_idx, val_idx = kfold.split(train_keys_combined).__next__()
                    train_keys = train_keys_combined[train_idx]
                    val_keys = train_keys_combined[val_idx]

                    splits.append(OrderedDict())
                    splits[-1]['train'] = train_keys
                    splits[-1]['val'] = val_keys
                    splits[-1]['test'] = test_keys
    
                save_pickle(splits, splits_file)

            else:
                self.print_to_log_file("Using splits from existing split file:", splits_file)
                splits = load_pickle(splits_file)
                self.print_to_log_file("The split file contains %d splits." % len(splits))

            self.print_to_log_file("Desired fold for training: %d" % self.fold)
            if self.fold < len(splits):
                tr_keys = splits[self.fold]['train']
                val_keys = splits[self.fold]['val']
                test_keys = splits[self.fold]['test']
                self.print_to_log_file("This split has %d training, %d validation, and %d test cases."
                                       % (len(tr_keys), len(val_keys), len(test_keys))
                                       )
            else:
                assert self.fold < len(splits), "You requested fold %d but splits file has %d splits" % (self.fold, len(splits))

        tr_keys.sort()
        val_keys.sort()
        test_keys.sort()
        self.dataset_tr = OrderedDict()
        for i in tr_keys:
            self.dataset_tr[i] = self.dataset[i]
        self.dataset_val = OrderedDict()
        for i in val_keys:
            self.dataset_val[i] = self.dataset[i]
        self.dataset_test = OrderedDict()
        for i in test_keys:
            self.dataset_test[i] = self.dataset[i]
    
    def update_fold(self, fold):
        """
        used to swap between folds for inference (ensemble of models from cross-validation)
        DO NOT USE DURING TRAINING AS THIS WILL NOT UPDATE THE DATASET SPLIT AND THE DATA AUGMENTATION GENERATORS
        :param fold:
        :return:
        """
        if fold is not None:
            if fold == "all":
                if self.output_folder.endswith("%s" % str(self.fold)):
                    self.output_folder = self.output_folder_base
                self.output_folder = join(self.output_folder, "%s" % str(fold))
            elif "domain" in str(fold):
                if self.output_folder.endswith("fold_%s" % str(self.fold)):
                    self.output_folder = self.output_folder_base
                self.output_folder = join(self.output_folder, "%s" % str(fold))
            else:
                if self.output_folder.endswith("fold_%s" % str(self.fold)):
                    self.output_folder = self.output_folder_base
                self.output_folder = join(self.output_folder, "fold_%s" % str(fold))
            self.fold = fold
    
    def get_basic_generators(self):
        self.load_dataset()
        self.do_split()

        if self.threeD:
            dl_tr = DataLoader3D(self.dataset_tr, self.basic_generator_patch_size, self.patch_size, self.batch_size,
                                 False, oversample_foreground_percent=self.oversample_foreground_percent,
                                 pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
            dl_val = DataLoader3D(self.dataset_val, self.patch_size, self.patch_size, self.batch_size, False,
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
        else:
            dl_tr = DataLoader2D(self.dataset_tr, self.basic_generator_patch_size, self.patch_size, self.batch_size,
                                 oversample_foreground_percent=self.oversample_foreground_percent,
                                 pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
            dl_val = DataLoader2D(self.dataset_val, self.patch_size, self.patch_size, self.batch_size,
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
        return dl_tr, dl_val

    def setup_DA_params(self):
        """
        - we increase roation angle from [-15, 15] to [-30, 30]
        - scale range is now (0.7, 1.4), was (0.85, 1.25)
        - we don't do elastic deformation anymore

        :return:
        """

        self.deep_supervision_scales = [[1, 1, 1]] + list(list(i) for i in 1 / np.cumprod(
            np.vstack(self.net_num_pool_op_kernel_sizes), axis=0))[:-1]

        if self.threeD:
            self.data_aug_params = default_3D_augmentation_params
            self.data_aug_params['rotation_x'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_y'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_z'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            if self.do_dummy_2D_aug:
                self.data_aug_params["dummy_2D"] = True
                self.print_to_log_file("Using dummy2d data augmentation")
                self.data_aug_params["elastic_deform_alpha"] = \
                    default_2D_augmentation_params["elastic_deform_alpha"]
                self.data_aug_params["elastic_deform_sigma"] = \
                    default_2D_augmentation_params["elastic_deform_sigma"]
                self.data_aug_params["rotation_x"] = default_2D_augmentation_params["rotation_x"]
        else:
            self.do_dummy_2D_aug = False
            if max(self.patch_size) / min(self.patch_size) > 1.5:
                default_2D_augmentation_params['rotation_x'] = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
            self.data_aug_params = default_2D_augmentation_params
        self.data_aug_params["mask_was_used_for_normalization"] = self.use_mask_for_norm

        if self.do_dummy_2D_aug:
            self.basic_generator_patch_size = get_patch_size(self.patch_size[1:],
                                                             self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])
            self.basic_generator_patch_size = np.array([self.patch_size[0]] + list(self.basic_generator_patch_size))
        else:
            self.basic_generator_patch_size = get_patch_size(self.patch_size, self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])

        self.data_aug_params["scale_range"] = (0.7, 1.4)
        self.data_aug_params["do_elastic"] = False
        self.data_aug_params['selected_seg_channels'] = [0]
        self.data_aug_params['patch_size_for_spatialtransform'] = self.patch_size

        self.data_aug_params["num_cached_per_thread"] = 2

    def maybe_update_lr(self, epoch=None):
        """
        if epoch is not None we overwrite epoch. Else we use epoch = self.epoch + 1

        (maybe_update_lr is called in on_epoch_end which is called before epoch is incremented.
        herefore we need to do +1 here)

        :param epoch:
        :return:
        """
        if epoch is None:
            ep = self.epoch + 1
        else:
            ep = epoch
        self.optimizer.param_groups[0]['lr'] = poly_lr(ep, self.max_num_epochs, self.initial_lr, 0.9)
        self.print_to_log_file("lr:", np.round(self.optimizer.param_groups[0]['lr'], decimals=6))

    def on_epoch_end(self):
        """
        overwrite patient-based early stopping. Always run to 1000 epochs
        :return:
        """
        super().on_epoch_end()
        continue_training = self.epoch < self.max_num_epochs

        # it can rarely happen that the momentum of nnUNetTrainerV2 is too high for some dataset. If at epoch 100 the
        # estimated validation Dice is still 0 then we reduce the momentum from 0.99 to 0.95
        if self.epoch == 100:
            if self.all_val_eval_metrics[-1] == 0:
                self.optimizer.param_groups[0]["momentum"] = 0.95
                self.network.apply(InitWeights_He(1e-2))
                self.print_to_log_file("At epoch 100, the mean foreground Dice was 0. This can be caused by a too "
                                       "high momentum. High momentum (0.99) is good for datasets where it works, but "
                                       "sometimes causes issues such as this one. Momentum has now been reduced to "
                                       "0.95 and network weights have been reinitialized")
        return continue_training

    def run_training(self):
        """
        if we run with -c then we need to set the correct lr for the first epoch, otherwise it will run the first
        continued epoch with self.initial_lr

        we also need to make sure deep supervision in the network is enabled for training, thus the wrapper
        :return:
        """
        self.maybe_update_lr(self.epoch)  # if we dont overwrite epoch then self.epoch+1 is used which is not what we
        # want at the start of the training
        ds = self.network.do_ds
        self.network.do_ds = True
        ret = super().run_training()
        self.network.do_ds = ds
        return ret

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        #### build bbox and points prompts
        iter_points, iter_bboxes = self.build_prompt_label(target[0][:, 0])
        # import pdb; pdb.set_trace()
        prompt_options = [[None, None], [iter_points, None],
                          [None, iter_bboxes], [iter_points, iter_bboxes]]

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

        if self.fp16:
            raise NotImplementedError("fp16 not implemented")
        self.optimizer.zero_grad()

        if do_backprop:
            for prompt in prompt_options:
                self.optimizer.zero_grad()
                output = self.network(data, prompt[0], prompt[1])
                l = self.loss(output, target)
                l.backward()
                self.optimizer.step()

        if run_online_evaluation:
            output = self.network(data, iter_points, iter_bboxes)
            l = self.loss(output, target)
            self.run_online_evaluation(output, target)

        del target, data

        return l.detach().cpu().numpy()
    
    def build_prompt_label(self, train_labels):
        bs = train_labels.shape[0]
        # generate prompt & label
        iter_bboxes = []
        iter_points_ax = []
        iter_point_labels = []
        for sample_idx in range(bs):
            # box prompt
            box = generate_box(train_labels[sample_idx])
            iter_bboxes.append(box)
            # point prompt
            num_positive_extra_max, num_negative_extra_max = 10, 10
            num_positive_extra = random.randint(0, num_positive_extra_max)
            num_negative_extra = random.randint(0, num_negative_extra_max)
            point, point_label = select_points(
                train_labels[sample_idx],
                num_positive_extra=num_positive_extra,
                num_negative_extra=num_negative_extra,
                fix_extra_point_num=num_positive_extra_max + num_negative_extra_max)
            iter_points_ax.append(point)
            iter_point_labels.append(point_label)
        # batched prompt
        iter_points_ax = torch.stack(iter_points_ax, dim=0).cuda()
        iter_point_labels = torch.stack(iter_point_labels, dim=0).cuda()
        iter_points = (iter_points_ax, iter_point_labels)
        iter_bboxes = torch.stack(iter_bboxes, dim=0).float().cuda()
        return iter_points, iter_bboxes
    
    def predict_with_prompt(self, data, points=None, boxes=None,):
        self.network.eval()
        self.network.do_ds = False
        with torch.no_grad():
            data = maybe_to_torch(data)
            if torch.cuda.is_available():
                data = to_cuda(data)
            output = self.network(data, points, boxes)
        
        # output = torch.argmax(output, dim=1, keepdim=False).squeeze(dim=0)
        output = output.squeeze(dim=0).detach().cpu().numpy()
        return output
    
    def predict_preprocessed_data_return_seg_and_softmax_withbox_points(self, data: np.ndarray, bbox=None, points=None,
                                                                        do_mirroring: bool = True,
                                                                        mirror_axes: Tuple[int] = None,
                                                                        use_sliding_window: bool = True, step_size: float = 0.5,
                                                                        use_gaussian: bool = True, pad_border_mode: str = 'constant',
                                                                        pad_kwargs: dict = None, all_in_gpu: bool = False,
                                                                        verbose: bool = True, mixed_precision: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param data:
        :param do_mirroring:
        :param mirror_axes:
        :param use_sliding_window:
        :param step_size:
        :param use_gaussian:
        :param pad_border_mode:
        :param pad_kwargs:
        :param all_in_gpu:
        :param verbose:
        :return:
        """
        if pad_border_mode == 'constant' and pad_kwargs is None:
            pad_kwargs = {'constant_values': 0}

        if do_mirroring and mirror_axes is None:
            mirror_axes = self.data_aug_params['mirror_axes']

        if do_mirroring:
            assert self.data_aug_params["do_mirror"], "Cannot do mirroring as test time augmentation when training " \
                                                      "was done without mirroring"

        valid = list((SegmentationNetwork, nn.DataParallel))
        assert isinstance(self.network, tuple(valid))

        current_mode = self.network.training
        do_ds = self.network.do_ds
        self.network.do_ds = False
        self.network.eval()
        ret = self.network.predict_3D_bbox_points(data, bbox, points, do_mirroring=do_mirroring, mirror_axes=mirror_axes,
                                                  use_sliding_window=use_sliding_window, step_size=step_size,
                                                  patch_size=self.patch_size, regions_class_order=self.regions_class_order,
                                                  use_gaussian=use_gaussian, pad_border_mode=pad_border_mode,
                                                  pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu, verbose=verbose,
                                                  mixed_precision=mixed_precision)
        self.network.train(current_mode)
        self.network.do_ds = do_ds
        return ret


def generate_box(pred_pre, bbox_shift=None):
    meaning_post_label = pred_pre # [h, w, d]
    ones_idx = (meaning_post_label > 0).nonzero(as_tuple=True)
    if all(tensor.nelement() == 0 for tensor in ones_idx):
        bboxes = torch.tensor([-1,-1,-1,-1,-1,-1])
        # print(bboxes, bboxes.shape)
        return bboxes
    min_coords = [dim.min() for dim in ones_idx]    # [x_min, y_min, z_min]
    max_coords = [dim.max() for dim in ones_idx]    # [x_max, y_max, z_max]

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
        corner_min = torch.tensor(corner_min)
        corner_max = torch.tensor(corner_max)
        return torch.cat((corner_min, corner_max), dim=0)
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
        corner_min = torch.tensor(corner_min)
        corner_max = torch.tensor(corner_max)
        return torch.cat((corner_min, corner_max), dim=0)


def select_points(preds, num_positive_extra=4, num_negative_extra=0, fix_extra_point_num=None):
    spacial_dim = 3
    points = torch.zeros((0, 3))
    labels = torch.zeros((0))
    pos_thred = 0.9
    neg_thred = 0.1
    
    # get pos/net indices
    positive_indices = torch.nonzero(preds > pos_thred, as_tuple=True) # ([pos x], [pos y], [pos z])
    negative_indices = torch.nonzero(preds < neg_thred, as_tuple=True)

    ones_idx = (preds > pos_thred).nonzero(as_tuple=True)
    if all(tmp.nelement() == 0 for tmp in ones_idx):
        # all neg
        num_positive_extra = 0
        selected_positive_point = torch.tensor([-1,-1,-1]).unsqueeze(dim=0)
        points = torch.cat((points, selected_positive_point), dim=0)
        labels = torch.cat((labels, torch.tensor([-1]).reshape(1)))
    else:
        # random select a pos point
        random_idx = torch.randint(len(positive_indices[0]), (1,))
        selected_positive_point = torch.tensor([positive_indices[i][random_idx] for i in range(spacial_dim)]).unsqueeze(dim=0)
        points = torch.cat((points, selected_positive_point), dim=0)
        labels = torch.cat((labels, torch.ones((1))))

    if num_positive_extra > 0:
        pos_idx_list = torch.randperm(len(positive_indices[0]))[:num_positive_extra]
        extra_positive_points = []
        for pos_idx in pos_idx_list:
            extra_positive_points.append([positive_indices[i][pos_idx] for i in range(spacial_dim)])
        extra_positive_points = torch.tensor(extra_positive_points).reshape(-1, 3)
        points = torch.cat((points, extra_positive_points), dim=0)
        labels = torch.cat((labels, torch.ones((extra_positive_points.shape[0]))))

    if num_negative_extra > 0:
        neg_idx_list = torch.randperm(len(negative_indices[0]))[:num_negative_extra]
        extra_negative_points = []
        for neg_idx in neg_idx_list:
            extra_negative_points.append([negative_indices[i][neg_idx] for i in range(spacial_dim)])
        extra_negative_points = torch.tensor(extra_negative_points).reshape(-1, 3)
        points = torch.cat((points, extra_negative_points), dim=0)
        labels = torch.cat((labels, torch.zeros((extra_negative_points.shape[0]))))
        # print('extra_negative_points ', extra_negative_points, extra_negative_points.shape)
        # print('==> points ', points.shape, labels)
    
    if fix_extra_point_num is None:
        left_point_num = num_positive_extra + num_negative_extra + 1 - labels.shape[0]
    else:
        left_point_num = fix_extra_point_num  + 1 - labels.shape[0]

    for _ in range(left_point_num):
        ignore_point = torch.tensor([-1,-1,-1]).unsqueeze(dim=0)
        points = torch.cat((points, ignore_point), dim=0)
        labels = torch.cat((labels, torch.tensor([-1]).reshape(1)))

    return (points, labels)
