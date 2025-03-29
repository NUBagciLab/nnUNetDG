import torch
from torch import Tensor, nn
from typing import Tuple, Type, Optional, List
from nnunet.network_architecture.generic_UNet import Generic_UNet, softmax_helper, InitWeights_He, ConvDropoutNormNonlin
from nnunet.network_architecture.custom_modules.prompt_encoder import nnPromptEncoder
from nnunet.network_architecture.custom_modules.transformer import TwoWayAttentionBlock, Attention

import numpy as np
import pdb
from torch.cuda.amp import autocast
from batchgenerators.augmentations.utils import pad_nd_image
from nnunet.utilities.random_stuff import no_op
from nnunet.utilities.to_torch import to_cuda, maybe_to_torch
from torch import nn
import torch
from scipy.ndimage.filters import gaussian_filter
from typing import Union, Tuple, List

class Generic_PromptUNet(Generic_UNet):
    def __init__(self, input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2, conv_op=nn.Conv2d,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, deep_supervision=True, dropout_in_localization=False,
                 final_nonlin=softmax_helper, weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None,
                 conv_kernel_sizes=None,
                 upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False,
                 max_num_features=None, basic_block=ConvDropoutNormNonlin,
                 seg_output_use_bias=False, transformer_dim=768, embedding_nums=3, text_embedding=None):
        super().__init__(input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage, feat_map_mul_on_downscale, conv_op, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, deep_supervision, dropout_in_localization, final_nonlin, weightInitializer, pool_op_kernel_sizes, conv_kernel_sizes, upscale_logits, convolutional_pooling, convolutional_upsampling, max_num_features, basic_block, seg_output_use_bias)
        self.transformer_dim = transformer_dim
        self.embedding_nums = embedding_nums

        self.cross_attention_list = []
        self.text_attention_list = []
        self.conv_in_list = []
        self.conv_out_list = []

        output_features = base_num_features
        output_features = int(np.round(output_features * (feat_map_mul_on_downscale**num_pool)))
        output_features = min(output_features, self.max_num_features)

        # the first conv reduces the number of features to match those of skip
        # the following convs work on that number of features
        # if not convolutional upsampling then the final conv reduces the num of features again
        u = self.embedding_nums - 1

        if not self.convolutional_upsampling:
            nfeatures_from_down = self.conv_blocks_context[-(2 + u)].output_channels
        else:
            nfeatures_from_down = self.conv_blocks_context[-(1 + u)].output_channels

        self.conv_in = nn.Conv3d(nfeatures_from_down, transformer_dim, kernel_size=1, stride=1, padding=0)
        self.cross_attention = TwoWayTransformer(depth=1,
                                            embedding_dim=transformer_dim,
                                            num_heads=8,
                                            mlp_dim=512)
        self.mask_token_num = min(nfeatures_from_down, 64)
        self.conv_out = nn.Sequential(nn.Conv3d(self.mask_token_num, nfeatures_from_down, kernel_size=1, stride=1, padding=0),
                                      norm_op(nfeatures_from_down),
                                      nonlin())
        
        self.iou_token = nn.Embedding(1, transformer_dim)
        self.mask_tokens = nn.Embedding(self.mask_token_num, transformer_dim)
        ### Add a tuning token to adjust the output, seems not needed
        # self.tuning_tokens = nn.Embedding(self.mask_token_num, transformer_dim)
        self.text_project = nn.Linear(768, self.mask_token_num)
        image_embedding_size = [8, 16, 16]
        patch_size = (32, 256, 256)
        self.prompt_encoder = nnPromptEncoder(embed_dim=transformer_dim,
                                              image_embedding_size=image_embedding_size,
                                              input_image_size=patch_size,
                                              embed_num=3)

        #### text embedding serves as the prompt for general cases
        #### register buffer to avoid being trained
        # self.register_buffer("text_embedding", text_embedding)
        if text_embedding is not None:
            self.text_embedding = nn.Parameter(text_embedding)
        else:
            print("No text embedding is provided, using random initialization")
            self.text_embedding = nn.Parameter(torch.randn((1, transformer_dim)))

    def forward(self, x, points=None, boxes=None,):
        # print("Input volume", torch.sum(x))
        image_pe = self.prompt_encoder.get_dense_pe()
        text_embedding = torch.repeat_interleave(self.text_embedding, x.size(0), dim=0).unsqueeze(1)
        # print(text_embedding.shape, x.shape)
        if self.training:
            ### text embedding might be None
            if np.random.rand() < 0.5:
                text_embedding = None
        sparse_prompt_embeddings = self.prompt_encoder(
            points=points,
            boxes=boxes,
            text_embedding=text_embedding,
        )
        skips = []
        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            skips.append(x)
            if not self.convolutional_pooling:
                x = self.td[d](x)

        x = self.conv_blocks_context[-1](x)
        
        # Concatenate output tokens
        # print(sparse_prompt_embeddings.shape)
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight,], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        seg_outputs = []
        for u in range(len(self.tu)):
            if u == (self.embedding_nums-1):
                x = self.conv_in(x)
                pe = image_pe
                
                x = self.cross_attention(image_embedding=x,
                                         image_pe=pe,
                                         context_embedding=tokens)
                x = x[:, 1:(self.mask_token_num+1)]
                ## N 1 C
                if text_embedding is not None:
                    text_embedding = text_embedding.squeeze(1)
                    text_project = self.text_project(text_embedding)
                    x = x + torch.einsum("bc,bchwd->bhwd", text_project, x).unsqueeze(1)

                x = self.conv_out(x)

            x = self.tu[u](x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            x = self.conv_blocks_localization[u](x)
            seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))
        
        if self._deep_supervision and self.do_ds:
            return tuple([seg_outputs[-1]] + [i(j) for i, j in
                                              zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
        else:
            # print("Total volume", torch.sum(seg_outputs[-1].argmax(1)))
            # print("Max value", torch.max(seg_outputs[-1]))
            return seg_outputs[-1]

    
    def predict_3D_bbox_points(self, x: np.ndarray, bbox: np.ndarray, points: Tuple[np.ndarray, np.ndarray], do_mirroring: bool, mirror_axes: Tuple[int, ...] = (0, 1, 2),
                   use_sliding_window: bool = False,
                   step_size: float = 0.5, patch_size: Tuple[int, ...] = None, regions_class_order: Tuple[int, ...] = None,
                   use_gaussian: bool = False, pad_border_mode: str = "constant",
                   pad_kwargs: dict = None, all_in_gpu: bool = False,
                   verbose: bool = True, mixed_precision: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Use this function to predict a 3D image. It does not matter whether the network is a 2D or 3D U-Net, it will
        detect that automatically and run the appropriate code.

        :param x: np.ndarray with shape (c_in, x, y, z)
        :return: (num_classes, x, y, z) np.ndarray with the prediction
        """
        torch.cuda.empty_cache()

        assert step_size <= 1, 'step_size must be smaller than 1. Otherwise there will be a gap between consecutive ' \
                               'predictions'

        if verbose: print("debug: mirroring", do_mirroring, "mirror_axes", mirror_axes)

        if pad_kwargs is None:
            pad_kwargs = {'constant_values': 0}

        # A very long time ago the mirror axes were (2, 3, 4) for a 3d network. This is just to intercept any old
        # code that uses this convention
        if len(mirror_axes):
            if self.conv_op == nn.Conv2d:
                if max(mirror_axes) > 1:
                    raise ValueError("mirror axes. duh")
            if self.conv_op == nn.Conv3d:
                if max(mirror_axes) > 2:
                    raise ValueError("mirror axes. duh")

        if self.training:
            print('WARNING! Network is in train mode during inference. This may be intended, or not...')

        assert len(x.shape) == 4, "data must have shape (c,x,y,z)"

        if mixed_precision:
            context = autocast
        else:
            context = no_op

        with context():
            with torch.no_grad():
                if self.conv_op == nn.Conv3d:
                    if use_sliding_window:
                        res = self._internal_predict_3D_3Dconv_tiled_boxes_points(x, bbox, points, step_size, do_mirroring, mirror_axes, patch_size,
                                                                     regions_class_order, use_gaussian, pad_border_mode,
                                                                     pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu,
                                                                     verbose=verbose)
                    else:
                        ### maske sure the x size is same as the patch size
                        if patch_size is not None:
                            assert x.shape[1:] == patch_size, "x shape must be the same as patch size"
                        res = self._internal_predict_3D_3Dconv(x, bbox, points, patch_size, do_mirroring, mirror_axes, regions_class_order,
                                                               pad_border_mode, pad_kwargs=pad_kwargs, verbose=verbose)
                else:
                    raise RuntimeError("Invalid conv op, cannot determine what dimensionality (2d/3d) the network is")

        return res
    
    @staticmethod
    def _compute_steps_for_sliding_window(patch_size: Tuple[int, ...], image_size: Tuple[int, ...], step_size: float) -> List[List[int]]:
        assert [i >= j for i, j in zip(image_size, patch_size)], "image size must be as large or larger than patch_size"
        assert 0 < step_size <= 1, 'step_size must be larger than 0 and smaller or equal to 1'

        # our step width is patch_size*step_size at most, but can be narrower. For example if we have image size of
        # 110, patch size of 64 and step_size of 0.5, then we want to make 3 steps starting at coordinate 0, 23, 46
        target_step_sizes_in_voxels = [i * step_size for i in patch_size]

        num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in zip(image_size, target_step_sizes_in_voxels, patch_size)]

        steps = []
        for dim in range(len(patch_size)):
            # the highest step value for this dimension is
            max_step_value = image_size[dim] - patch_size[dim]
            if num_steps[dim] > 1:
                actual_step_size = max_step_value / (num_steps[dim] - 1)
            else:
                actual_step_size = 99999999999  # does not matter because there is only one step at 0

            steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]

            steps.append(steps_here)

        return steps
    
    def _internal_predict_3D_3Dconv_tiled_boxes_points(self, x: np.ndarray, bbox: np.ndarray, points: np.ndarray, step_size: float, do_mirroring: bool, mirror_axes: tuple,
                                          patch_size: tuple, regions_class_order: tuple, use_gaussian: bool,
                                          pad_border_mode: str, pad_kwargs: dict, all_in_gpu: bool,
                                          verbose: bool) -> Tuple[np.ndarray, np.ndarray]:
        # better safe than sorry
        assert len(x.shape) == 4, "x must be (c, x, y, z)"

        if verbose: print("step_size:", step_size)
        if verbose: print("do mirror:", do_mirroring)

        assert patch_size is not None, "patch_size cannot be None for tiled prediction"
        data = x
        data_shape = data.shape
        # compute the steps for sliding window
        steps = self._compute_steps_for_sliding_window(patch_size, data_shape[1:], step_size)
        num_tiles = len(steps[0]) * len(steps[1]) * len(steps[2])

        if verbose:
            print("data shape:", data_shape)
            print("patch size:", patch_size)
            print("steps (x, y, and z):", steps)
            print("number of tiles:", num_tiles)

        # we only need to compute that once. It can take a while to compute this due to the large sigma in
        # gaussian_filter
        if use_gaussian and num_tiles > 1:
            if self._gaussian_3d is None or not all(
                    [i == j for i, j in zip(patch_size, self._patch_size_for_gaussian_3d)]):
                if verbose: print('computing Gaussian')
                gaussian_importance_map = self._get_gaussian(patch_size, sigma_scale=1. / 8)

                self._gaussian_3d = gaussian_importance_map
                self._patch_size_for_gaussian_3d = patch_size
                if verbose: print("done")
            else:
                if verbose: print("using precomputed Gaussian")
                gaussian_importance_map = self._gaussian_3d

            gaussian_importance_map = torch.from_numpy(gaussian_importance_map)

            #predict on cpu if cuda not available
            if torch.cuda.is_available():
                gaussian_importance_map = gaussian_importance_map.cuda(self.get_device(), non_blocking=True)

        else:
            gaussian_importance_map = None

        if all_in_gpu:
            # If we run the inference in GPU only (meaning all tensors are allocated on the GPU, this reduces
            # CPU-GPU communication but required more GPU memory) we need to preallocate a few things on GPU

            if use_gaussian and num_tiles > 1:
                # half precision for the outputs should be good enough. If the outputs here are half, the
                # gaussian_importance_map should be as well
                gaussian_importance_map = gaussian_importance_map.half()

                # make sure we did not round anything to 0
                gaussian_importance_map[gaussian_importance_map == 0] = gaussian_importance_map[
                    gaussian_importance_map != 0].min()

                add_for_nb_of_preds = gaussian_importance_map
            else:
                add_for_nb_of_preds = torch.ones(patch_size, device=self.get_device())

            if verbose: print("initializing result array (on GPU)")
            aggregated_results = torch.zeros([self.num_classes] + list(data.shape[1:]), dtype=torch.half,
                                             device=self.get_device())

            if verbose: print("moving data to GPU")
            data = torch.from_numpy(data).cuda(self.get_device(), non_blocking=True)

            if verbose: print("initializing result_numsamples (on GPU)")
            aggregated_nb_of_predictions = torch.zeros([self.num_classes] + list(data.shape[1:]), dtype=torch.half,
                                                       device=self.get_device())

        else:
            if use_gaussian and num_tiles > 1:
                add_for_nb_of_preds = self._gaussian_3d
            else:
                add_for_nb_of_preds = np.ones(patch_size, dtype=np.float32)
            aggregated_results = np.zeros([self.num_classes] + list(data.shape[1:]), dtype=np.float32)
            aggregated_nb_of_predictions = np.zeros([self.num_classes] + list(data.shape[1:]), dtype=np.float32)

        current_bbox = None
        current_points = None

        def check_bbox(x_min, x_max, t_min, t_max):
            ### check if [x_min, x_max] and [t_min, t_max] has overlap with each other
            # if x_min >= t_min or x_max <= t_max:
            #     return True
            # return False
            if x_max <= t_min or x_min >= t_max:
                return False
            return True

        mark_background = False
        for x in steps[0]:
            lb_x = x
            ub_x = x + patch_size[0]
            for y in steps[1]:
                lb_y = y
                ub_y = y + patch_size[1]
                for z in steps[2]:
                    lb_z = z
                    ub_z = z + patch_size[2]
                    mark_background = False
                    current_bbox  = None
                    if bbox is not None:
                        x_min, y_min, z_min, x_max, y_max, z_max = bbox[0]
                        ### check if the bbox has overlap with the patch
                        if check_bbox(x_min, x_max, lb_x, ub_x) and \
                            check_bbox(y_min, y_max, lb_y, ub_y) and \
                            check_bbox(z_min, z_max, lb_z, ub_z):
                            current_x_min = max(x_min-lb_x, 0)
                            current_x_max = min(x_max-lb_x, patch_size[0])
                            current_y_min = max(y_min-lb_y, 0)
                            current_y_max = min(y_max-lb_y, patch_size[1])
                            current_z_min = max(z_min-lb_z, 0)
                            current_z_max = min(z_max-lb_z, patch_size[2])
                            current_bbox = np.array([current_x_min, current_y_min, current_z_min, current_x_max, current_y_max, current_z_max])
                            current_bbox = np.expand_dims(current_bbox, axis=0)
                        else:
                            ### if there is no overlap, then this patch is not useful
                            mark_background = True
                    
                    current_in_points = None
                    current_points, current_points_label = None, None
                    if points is not None:
                        points_pos = points[0][0]
                        for i in range(points_pos.shape[0]):
                            x, y, z = points_pos[i]
                            
                            if x >= lb_x and x < ub_x and y >= lb_y and y < ub_y and z >= lb_z and z < ub_z:
                                current_point = points_pos[i]-np.array([lb_x, lb_y, lb_z])
                                if current_points is None:
                                    current_points = [current_point,]
                                    current_points_label = [points[1][0][i]]
                                else:
                                    current_points.append(current_point)
                                    current_points_label.append(points[1][0][i])
                        
                        if current_points is not None:
                            current_points = np.stack(current_points, axis=0)
                            current_points_label = np.array(current_points_label)
                            current_in_points = (np.expand_dims(current_points, axis=0),
                                                 np.expand_dims(current_points_label, axis=0))
                            ### check if all points are negative
                            if np.sum(current_points_label) == 0:
                                current_in_points = None
                    # import pdb; pdb.set_trace()

                    predicted_patch = self._internal_maybe_mirror_and_pred_3D_bbox_points(
                        data[None, :, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z], 
                        current_bbox, current_in_points, mirror_axes, do_mirroring,
                        gaussian_importance_map)[0]
                    
                    if mark_background:
                        ## assert the predicted patch is background
                        predicted_patch[1:] = 0

                    # if mark_background:
                    #     predicted_patch = np.zeros_like(predicted_patch)

                    if all_in_gpu:
                        predicted_patch = predicted_patch.half()
                    else:
                        predicted_patch = predicted_patch.cpu().numpy()

                    aggregated_results[:, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += predicted_patch
                    if not mark_background:
                        aggregated_nb_of_predictions[:, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += add_for_nb_of_preds

        # computing the class_probabilities by dividing the aggregated result with result_numsamples
        aggregated_results /= aggregated_nb_of_predictions
        del aggregated_nb_of_predictions

        if regions_class_order is None:
            predicted_segmentation = aggregated_results.argmax(0)
        else:
            if all_in_gpu:
                class_probabilities_here = aggregated_results.detach().cpu().numpy()
            else:
                class_probabilities_here = aggregated_results
            predicted_segmentation = np.zeros(class_probabilities_here.shape[1:], dtype=np.float32)
            for i, c in enumerate(regions_class_order):
                predicted_segmentation[class_probabilities_here[i] > 0.5] = c

        if all_in_gpu:
            if verbose: print("copying results to CPU")

            if regions_class_order is None:
                predicted_segmentation = predicted_segmentation.detach().cpu().numpy()

            aggregated_results = aggregated_results.detach().cpu().numpy()

        if verbose: print("prediction done")
        return predicted_segmentation, aggregated_results
    
    
    def _internal_maybe_mirror_and_pred_3D_bbox_points(self, x: Union[np.ndarray, torch.tensor], 
                                           bbox,
                                           points,
                                           mirror_axes: tuple,
                                           do_mirroring: bool = True,
                                           mult= None) -> torch.tensor:
        assert len(x.shape) == 5, 'x must be (b, c, x, y, z)'

        # if cuda available:
        #   everything in here takes place on the GPU. If x and mult are not yet on GPU this will be taken care of here
        #   we now return a cuda tensor! Not numpy array!

        x = maybe_to_torch(x)
        if points is not None:
            points = [maybe_to_torch(point) for point in points]
    
        if bbox is not None:
            bbox = maybe_to_torch(bbox)
        result_torch = torch.zeros([1, self.num_classes] + list(x.shape[2:]),
                                   dtype=torch.float)

        if torch.cuda.is_available():
            x = to_cuda(x, gpu_id=self.get_device())
            if points is not None:
                points = [to_cuda(point, gpu_id=self.get_device()) for point in points]
        
            if bbox is not None:
                bbox = to_cuda(bbox, gpu_id=self.get_device())
            result_torch = result_torch.cuda(self.get_device(), non_blocking=True)

        if mult is not None:
            mult = maybe_to_torch(mult)
            if torch.cuda.is_available():
                mult = to_cuda(mult, gpu_id=self.get_device())

        if do_mirroring:
            mirror_idx = 8
            num_results = 2 ** len(mirror_axes)
        else:
            mirror_idx = 1
            num_results = 1

        for m in range(mirror_idx):
            if m == 0:
                pred = self.inference_apply_nonlin(self(x, points, bbox))
                result_torch += 1 / num_results * pred

            if m == 1 and (2 in mirror_axes):
                pred = self.inference_apply_nonlin(self(torch.flip(x, (4, )), points, bbox))
                result_torch += 1 / num_results * torch.flip(pred, (4,))

            if m == 2 and (1 in mirror_axes):
                pred = self.inference_apply_nonlin(self(torch.flip(x, (3, )), points, bbox))
                result_torch += 1 / num_results * torch.flip(pred, (3,))

            if m == 3 and (2 in mirror_axes) and (1 in mirror_axes):
                pred = self.inference_apply_nonlin(self(torch.flip(x, (4, 3)), points, bbox))
                result_torch += 1 / num_results * torch.flip(pred, (4, 3))

            if m == 4 and (0 in mirror_axes):
                pred = self.inference_apply_nonlin(self(torch.flip(x, (2, )), points, bbox))
                result_torch += 1 / num_results * torch.flip(pred, (2,))

            if m == 5 and (0 in mirror_axes) and (2 in mirror_axes):
                pred = self.inference_apply_nonlin(self(torch.flip(x, (4, 2)), points, bbox))
                result_torch += 1 / num_results * torch.flip(pred, (4, 2))

            if m == 6 and (0 in mirror_axes) and (1 in mirror_axes):
                pred = self.inference_apply_nonlin(self(torch.flip(x, (3, 2)), points, bbox))
                result_torch += 1 / num_results * torch.flip(pred, (3, 2))

            if m == 7 and (0 in mirror_axes) and (1 in mirror_axes) and (2 in mirror_axes):
                pred = self.inference_apply_nonlin(self(torch.flip(x, (4, 3, 2)), points, bbox))
                result_torch += 1 / num_results * torch.flip(pred, (4, 3, 2))

        if mult is not None:
            result_torch[:, :] *= mult

        return result_torch



class TwoWayTransformer(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )
        
        
        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        context_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x embedding_dim x h x w for any h and w.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          context_embedding (torch.Tensor): the embedding to add to the query contexts.
            Must have shape B x N_contexts x embedding_dim for any N_contexts.

        Returns:
          torch.Tensor: the processed image_embedding
        """
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, h, w, d = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)
        # Prepare queries
        queries = context_embedding
        keys = image_embedding

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=context_embedding,
                key_pe=image_pe,
            )

        # Apply the final attention layer from the points to the image
        q = queries + context_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        ### key B hwd C, query B Np C
        out = torch.einsum("bhc,bnc->bhn", keys, queries)
        ## transpose back to B x C x H x W
        out = out.permute(0, 2, 1).reshape(bs, -1, h, w, d)

        return out


if __name__ == "__main__":
    num_input_channels = 1
    base_num_features = 16
    num_classes = 2
    conv_op = nn.Conv3d
    dropout_op = nn.Dropout3d
    norm_op = nn.InstanceNorm3d
    norm_op_kwargs = {'eps': 1e-5, 'affine': True}
    dropout_op_kwargs = {'p': 0, 'inplace': True}
    net_nonlin = nn.LeakyReLU
    net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
    conv_per_stage = 2
    patch_size = (32, 256, 256)
    transformer_dim = 768
    net_conv_kernel_sizes = [[1, 3, 3],
                            [3, 3, 3],
                            [3, 3, 3],
                            [3, 3, 3],
                            [3, 3, 3],
                            [3, 3, 3],
                            [3, 3, 3],]
    net_num_pool_op_kernel_sizes = [[1, 2, 2],
                                   [1, 2, 2],
                                   [2, 2, 2],
                                   [2, 2, 2],
                                   [2, 2, 2],
                                   [2, 2, 2],] ## [2, 4, 4]
    net_numpool = len(net_num_pool_op_kernel_sizes)
    image_embedding_size = [8, 16, 16]
    text_embedding = torch.randn((1, transformer_dim))
    decoder = Generic_PromptUNet(num_input_channels, base_num_features, num_classes, net_numpool,
                            conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                            dropout_op_kwargs,
                            net_nonlin, net_nonlin_kwargs, False, False, lambda x: x, InitWeights_He(1e-2),
                            net_num_pool_op_kernel_sizes, net_conv_kernel_sizes, False, True, True,
                            transformer_dim=transformer_dim, embedding_nums=3,
                            text_embedding=text_embedding)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder.to(device)
    decoder.eval()
    for i  in range(30):
        image = torch.randn((2, 1, 32, 256, 256)).to(device)
        n_points = 3
        points = torch.randn((2, n_points, 3)).to(device)
        labels = torch.ones((2, n_points)).to(device)
        input_points = (points, labels)
        output = decoder(image, points=input_points)
        print(output.shape)
