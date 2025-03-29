# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F
from monai.networks.nets import ViT

from typing import List, Tuple, Type, Optional
from monai.networks.nets.unetr import UnetrUpBlock, UnetOutBlock, UnetrPrUpBlock

from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.network_architecture.neural_network import SegmentationNetwork
# from segment_anything_volumetric.modeling.transformer import TwoWayTransformer
TwoWayTransformer = None

class UNETR(SegmentationNetwork):
    def __init__(
        self,
        in_channels: int=1,
        out_channels: int=2,
        image_size=(32, 256, 256),
        patch_size=(4, 16, 16),
        transformer_dim: int=768,
        mlp_dim: int=3072,
        num_layers: int=12,
        num_heads: int=12,
        pos_embed = 'perceptron',
        num_hidden_features=3,
        feature_size=96,
        norm_name='instance',
        final_nonlin=softmax_helper,
        _deep_supervision=True,
        do_ds=True,
    ):
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.image_encoder=ViT(
                        in_channels=in_channels,
                        img_size=image_size,
                        patch_size=patch_size,
                        hidden_size=transformer_dim,
                        mlp_dim=mlp_dim,
                        num_layers=num_layers,
                        num_heads=num_heads,
                        pos_embed=pos_embed,
                        classification=False,
                    )
        self.transformer_dim = transformer_dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.bottom_feature_size = [s//si for s, si in zip(image_size, patch_size)]

        # self.iou_token = nn.Embedding(1, transformer_dim)
        # self.num_mask_tokens = num_multimask_outputs + 1
        # self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        # self.transformer = TwoWayTransformer(
        #     depth=2,
        #     embedding_dim=transformer_dim,
        #     mlp_dim=2048,
        #     num_heads=8,
        # )

        self.num_hidden_features = num_hidden_features
        self.feature_size = feature_size
        self.bottom_block = nn.Conv3d(transformer_dim, feature_size, kernel_size=3, stride=1, padding=1)
        self.bottom_up = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=True)
        self.decoder_list = nn.ModuleList([])
        self.encoder_list = nn.ModuleList([])
        self.norm_name = norm_name
        self.output_scale = []
        last_feature_size = [s//sz for s, sz in zip(self.patch_size, (1, 2, 2))]
        self.do_ds = do_ds

        for i in range(self.num_hidden_features):
            if i < 2:
                self.encoder_list.append(UnetrPrUpBlock(spatial_dims=3,
                                                        in_channels=transformer_dim,
                                                        out_channels=feature_size,
                                                        num_layer=i,
                                                        kernel_size=3,
                                                        stride=1,
                                                        upsample_kernel_size=2,
                                                        norm_name=norm_name))
                self.decoder_list.append(UnetrUpBlock(spatial_dims=3,
                                                        in_channels=feature_size,
                                                        out_channels=feature_size,
                                                        kernel_size=3,
                                                        upsample_kernel_size=2,
                                                        norm_name=norm_name))
            else:
                self.encoder_list.append(nn.Sequential(UnetrPrUpBlock(spatial_dims=3,
                                                                    in_channels=transformer_dim,
                                                                    out_channels=feature_size,
                                                                    num_layer=1,
                                                                    kernel_size=3,
                                                                    stride=1,
                                                                    upsample_kernel_size=2,
                                                                    norm_name=norm_name),
                                                       UnetrPrUpBlock(spatial_dims=3,
                                                                      in_channels=feature_size,
                                                                      out_channels=feature_size,
                                                                      num_layer=(i-2),
                                                                      kernel_size=3,
                                                                      upsample_kernel_size=(1, 2, 2),
                                                                      stride=1,
                                                                      norm_name=norm_name)))
                self.decoder_list.append(UnetrUpBlock(spatial_dims=3,
                                                        in_channels=feature_size,
                                                        out_channels=feature_size,
                                                        kernel_size=3,
                                                        upsample_kernel_size=(1, 2, 2),
                                                        norm_name=norm_name))

            last_feature_size = [max(s//2, 1) for s in last_feature_size]
            self.output_scale.append(tuple(last_feature_size))

        self._deep_supervision = _deep_supervision
        self.final_nonlin = final_nonlin

        # self.upscale_logits_ops = []
        self.seg_outputs = nn.ModuleList([])
        for i in range(self.num_hidden_features):
            # if i < self.num_hidden_features - 1:
            #   self.upscale_logits_ops.append(nn.Upsample(scale_factor=self.output_scale[i],
            #                                             mode='trilinear', align_corners=True))
            self.seg_outputs.append(UnetOutBlock(spatial_dims=3,
                                              in_channels=feature_size,
                                              out_channels=out_channels))
        
    def forward(
        self,
        images,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Returns:
          torch.Tensor: batched predicted masks
        """

        image_embeddings, hidden_status = self.image_encoder(images)
        shallow_features = [hidden_status[9], hidden_status[6], hidden_status[3]]
        #### transpose the image embeddings back to the original size
        image_embeddings = image_embeddings.permute(0, 1, 2).reshape(images.shape[0], self.transformer_dim, *self.bottom_feature_size)
        shallow_features = [i.permute(0, 1, 2).reshape(images.shape[0], self.transformer_dim, *self.bottom_feature_size) for i in shallow_features]
      
        upscaled_embedding = self.bottom_block(image_embeddings)
        upscaled_embedding = self.bottom_up(upscaled_embedding)

        shallow_features = [self.bottom_up(i) for i in shallow_features]

        ### decoder
        seg_outputs = []
        for i in range(self.num_hidden_features):
            shallow_feature = shallow_features[i]
            ### upscale the sallow feature
            shallow_feature = self.encoder_list[i](shallow_feature)
            upscaled_embedding = self.decoder_list[i](upscaled_embedding, shallow_feature)
            seg_outputs.append(self.final_nonlin(self.seg_outputs[i](upscaled_embedding)))

        if self._deep_supervision and self.do_ds:
            return tuple([seg_outputs[-1]] + seg_outputs[:-1][::-1])
        else:
            return seg_outputs[-1]


if __name__ == '__main__':
    ### Generate the test cases
    image_encoder_type = 'vit'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    unetr = UNETR(in_channels=1, out_channels=2, image_size=(32, 256, 256), patch_size=(4, 16, 16))
    unetr = unetr.to(device)
    test_image = torch.randn(2, 1, 32, 256, 256).to(device)
    test_output = unetr(test_image)
    for i in test_output:
        print(i.shape)
