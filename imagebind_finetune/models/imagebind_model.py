##!/usr/bin/env python3
## Portions Copyright (c) Meta Platforms, Inc. and affiliates.
## All rights reserved.

import os
from functools import partial
from types import SimpleNamespace
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F 

from imagebind_finetune.models.helpers import (EinOpsRearrange, LearnableLogitScaling, Normalize,
                            SelectElement, SelectEOSAndProject)
from imagebind_finetune.models.multimodal_preprocessors import (AudioPreprocessor,
                                             IMUPreprocessor, PadIm2Video,
                                             PatchEmbedGeneric,
                                             RGBDTPreprocessor,
                                             SpatioTemporalPosEmbeddingHelper,
                                             TextPreprocessor,
                                             ThermalPreprocessor)
from imagebind_finetune.models.transformer import MultiheadAttention, SimpleTransformer

from layers.temporal_av_attn_layer import TemporalAttentionModule, TemporalLinearModule

import pdb

ModalityType = SimpleNamespace(
    VISION="vision",
    TEXT="text",
    AUDIO="audio",
    THERMAL="thermal",
    DEPTH="depth",
    IMU="imu",
)

## [Novelty Module] Scheme 2: Semantic Manifold Alignment Adapter (AMA)

class AttentiveManifoldAdapter(nn.Module):
    def __init__(self, embed_dim, num_tokens=4, hidden_dim=512):
        super().__init__()
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim
        
        self.attn_pool = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.Softmax(dim=0)
        )
        
        self.visual_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )

        self.audio_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )

        self.other_tokens_num = num_tokens - 2
        if self.other_tokens_num > 0:
            self.context_head = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, self.other_tokens_num * embed_dim)
            )
            
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=4, 
            dim_feedforward=hidden_dim,
            dropout=0.1,
            activation='gelu',
            batch_first=False 
        )
        self.interaction_layer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        self.gate = nn.Parameter(torch.tensor([-2.0])) 
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, text_tokens):
        attn_weights = self.attn_pool(text_tokens)
        text_sig = (text_tokens * attn_weights).sum(dim=0)
        
        shared_feat = text_sig
        
        p_visual = self.visual_head(shared_feat).unsqueeze(0)
        p_audio = self.audio_head(shared_feat).unsqueeze(0)
        
        prompts_list = [p_visual, p_audio]
        
        if self.other_tokens_num > 0:
            p_context = self.context_head(shared_feat) 
            p_context = p_context.view(-1, self.other_tokens_num, self.embed_dim).permute(1, 0, 2)
            prompts_list.append(p_context)
            
        raw_prompts = torch.cat(prompts_list, dim=0)
        refined_prompts = self.interaction_layer(raw_prompts)
        text_anchor = text_sig.unsqueeze(0).expand(self.num_tokens, -1, -1)
        final_prompts = text_anchor + torch.sigmoid(self.gate) * refined_prompts
        
        return self.dropout(self.norm(final_prompts))

## ImageBind Main Model

class ImageBindModel(nn.Module):
    def __init__(
        self,
        video_frames=2,
        kernel_size=(2, 14, 14),
        audio_kernel_size=16,
        audio_stride=10,
        out_embed_dim=768,
        vision_embed_dim=1024,
        vision_num_blocks=24,
        vision_num_heads=16,
        audio_embed_dim=768,
        audio_num_blocks=12,
        audio_num_heads=12,
        audio_num_mel_bins=128,
        audio_target_len=204,
        audio_drop_path=0.1,
        text_embed_dim=768,
        text_num_blocks=12,
        text_num_heads=12,
        depth_embed_dim=384,
        depth_kernel_size=16,
        depth_num_blocks=12,
        depth_num_heads=8,
        depth_drop_path=0.0,
        thermal_embed_dim=768,
        thermal_kernel_size=16,
        thermal_num_blocks=12,
        thermal_num_heads=12,
        thermal_drop_path=0.0,
        imu_embed_dim=512,
        imu_kernel_size=8,
        imu_num_blocks=6,
        imu_num_heads=8,
        imu_drop_path=0.7,
        spatial_av_attn_layer_ids=([], []),
        sattn_flag='none',
        tattn_flag=False,
        sa_layer_num=1,
        xa_layer_num=1,
        feat_dim=1024,
        hid_dim=256,
        d_ff=512,
        head_num=1,
        dropout=0.1,
        use_adj_in_attn=True,
        gamma=0.6,
        bias=0.2,
        use_mask_in_attn=True,
        win_size=4,
        norm_flag=None,
        text_tune_flag=False,
        manifold_align_flag=False,
        manifold_token_num=4,
        use_cluster_module=False,
        semantic_ablation_v='none',
        semantic_ablation_a='none',
        semantic_noise_scale_v=1.0,
        semantic_noise_scale_a=1.0,
    ):
        super().__init__()

        self.modality_preprocessors = self._create_modality_preprocessors(
            video_frames, vision_embed_dim, kernel_size, text_embed_dim,
            audio_embed_dim, audio_kernel_size, audio_stride, audio_num_mel_bins, audio_target_len,
            depth_embed_dim, depth_kernel_size, thermal_embed_dim, thermal_kernel_size, imu_embed_dim,
        )

        self.modality_trunks = self._create_modality_trunks(
            vision_embed_dim, vision_num_blocks, vision_num_heads,
            text_embed_dim, text_num_blocks, text_num_heads,
            audio_embed_dim, audio_num_blocks, audio_num_heads, audio_drop_path,
            depth_embed_dim, depth_num_blocks, depth_num_heads, depth_drop_path,
            thermal_embed_dim, thermal_num_blocks, thermal_num_heads, thermal_drop_path,
            imu_embed_dim, imu_num_blocks, imu_num_heads, imu_drop_path,
        )

        self.modality_heads = self._create_modality_heads(
            out_embed_dim, vision_embed_dim, text_embed_dim, audio_embed_dim,
            depth_embed_dim, thermal_embed_dim, imu_embed_dim,
        )

        self.modality_postprocessors = self._create_modality_postprocessors(out_embed_dim)

        self.spatial_av_attn_layer_ids = spatial_av_attn_layer_ids
        self.sattn_flag = sattn_flag
        if self.sattn_flag != 'none':
            self.spatial_av_layers = self._create_spatial_av_layers(
                spatial_av_attn_layer_ids, audio_embed_dim, vision_embed_dim,
            )

        self.tattn_flag = tattn_flag
        if self.tattn_flag:
            self.temporal_av_layer = TemporalAttentionModule(
                sa_layer_num, xa_layer_num, feat_dim, hid_dim, d_ff, head_num,
                dropout, use_adj_in_attn, gamma, bias, use_mask_in_attn, win_size, norm_flag
            )
        
        self.text_tune_flag = text_tune_flag
        if self.text_tune_flag:
            self.task_res_text_learner = nn.Linear(text_embed_dim, feat_dim, bias=False)
            self.task_res_alpha = nn.Parameter(torch.FloatTensor([float('-inf')])) 

        self.manifold_align_flag = manifold_align_flag
        self.manifold_token_num = manifold_token_num
        if self.manifold_align_flag:
            self.manifold_adapter = AttentiveManifoldAdapter(
                embed_dim=text_embed_dim, num_tokens=manifold_token_num, hidden_dim=text_embed_dim // 2 
            )

        self.use_cluster_module = use_cluster_module
        self.semantic_ablation_v = semantic_ablation_v
        self.semantic_ablation_a = semantic_ablation_a
        self.semantic_noise_scale_v = semantic_noise_scale_v
        self.semantic_noise_scale_a = semantic_noise_scale_a
        
        if self.use_cluster_module:
## === Video Cluster Module ===
            self.cluster_proj = nn.Sequential(
                nn.Linear(out_embed_dim, out_embed_dim),
                nn.GELU(),
                nn.Linear(out_embed_dim, out_embed_dim)
            )
            self._init_mlp_kaiming(self.cluster_proj)

            self.cross_attn = nn.MultiheadAttention(
                embed_dim=feat_dim, num_heads=4, dropout=dropout, batch_first=True
            )
            self.cross_attn_dropout = nn.Dropout(dropout)
            self.norm_video = nn.LayerNorm(feat_dim)
            
## === Audio Cluster Module ===
            self.audio_cluster_proj = nn.Sequential(
                nn.Linear(out_embed_dim, out_embed_dim),
                nn.GELU(),
                nn.Linear(out_embed_dim, out_embed_dim)
            )
            self._init_mlp_kaiming(self.audio_cluster_proj)

            self.audio_cross_attn = nn.MultiheadAttention(
                embed_dim=feat_dim, num_heads=4, dropout=dropout, batch_first=True
            )
            self.audio_cross_attn_dropout = nn.Dropout(dropout)
            self.norm_audio = nn.LayerNorm(feat_dim)
            
            self._init_cross_attn_near_zero()

##  [: ]
## alpha beta
        self.alpha_v = nn.Parameter(torch.tensor(1.0))
        self.beta_v  = nn.Parameter(torch.tensor(0.0))
        self.alpha_a = nn.Parameter(torch.tensor(1.0))
        self.beta_a  = nn.Parameter(torch.tensor(0.0))

    def _init_mlp_kaiming(self, module):
        for m in module:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _init_mlp_as_identity(self, module):
        for m in module:
            if isinstance(m, nn.Linear):
                nn.init.eye_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def _init_cross_attn_near_zero(self):
## Video
        nn.init.zeros_(self.cross_attn.out_proj.weight)
        nn.init.zeros_(self.cross_attn.out_proj.bias)
## Audio
        nn.init.zeros_(self.audio_cross_attn.out_proj.weight)
        nn.init.zeros_(self.audio_cross_attn.out_proj.bias)

    def _create_spatial_av_layers(self, spatial_av_attn_layer_ids, audio_embed_dim=768, vision_embed_dim=1024):
        audio_lids, vision_lids = spatial_av_attn_layer_ids
        module_a2v = nn.Linear(audio_embed_dim, vision_embed_dim, bias=False)
        module_v2a = nn.Linear(vision_embed_dim, audio_embed_dim, bias=False)
        layers_a2v = nn.ModuleList([copy.deepcopy(module_a2v) for i in range(len(audio_lids))])
        layers_v2a = nn.ModuleList([copy.deepcopy(module_v2a) for i in range(len(vision_lids))])
        return nn.ModuleDict({ModalityType.AUDIO: layers_a2v, ModalityType.VISION: layers_v2a})

    def _create_modality_preprocessors(self, video_frames=2, vision_embed_dim=1024, kernel_size=(2, 14, 14),
                                       text_embed_dim=768, audio_embed_dim=768, audio_kernel_size=16,
                                       audio_stride=10, audio_num_mel_bins=128, audio_target_len=204,
                                       depth_embed_dim=768, depth_kernel_size=16, thermal_embed_dim=768,
                                       thermal_kernel_size=16, imu_embed_dim=512):
        rgbt_stem = PatchEmbedGeneric(
            proj_stem=[
                PadIm2Video(pad_type="repeat", ntimes=2),
                nn.Conv3d(in_channels=3, kernel_size=kernel_size, out_channels=vision_embed_dim, stride=kernel_size, bias=False),
            ]
        )
        rgbt_preprocessor = RGBDTPreprocessor(
            img_size=[3, video_frames, 224, 224], num_cls_tokens=1,
            pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
            rgbt_stem=rgbt_stem, depth_stem=None,
        )
        text_preprocessor = TextPreprocessor(context_length=77, vocab_size=49408, embed_dim=text_embed_dim, causal_masking=True)
        audio_stem = PatchEmbedGeneric(
            proj_stem=[nn.Conv2d(in_channels=1, kernel_size=audio_kernel_size, stride=audio_stride, out_channels=audio_embed_dim, bias=False)],
            norm_layer=nn.LayerNorm(normalized_shape=audio_embed_dim),
        )
        audio_preprocessor = AudioPreprocessor(
            img_size=[1, audio_num_mel_bins, audio_target_len], num_cls_tokens=1,
            pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True), audio_stem=audio_stem,
        )
        depth_stem = PatchEmbedGeneric(
            [nn.Conv2d(in_channels=1, kernel_size=depth_kernel_size, out_channels=depth_embed_dim, stride=depth_kernel_size, bias=False)],
            norm_layer=nn.LayerNorm(normalized_shape=depth_embed_dim),
        )
        depth_preprocessor = RGBDTPreprocessor(
            img_size=[1, 224, 224], num_cls_tokens=1,
            pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True), rgbt_stem=None, depth_stem=depth_stem,
        )
        thermal_stem = PatchEmbedGeneric(
            [nn.Conv2d(in_channels=1, kernel_size=thermal_kernel_size, out_channels=thermal_embed_dim, stride=thermal_kernel_size, bias=False)],
            norm_layer=nn.LayerNorm(normalized_shape=thermal_embed_dim),
        )
        thermal_preprocessor = ThermalPreprocessor(
            img_size=[1, 224, 224], num_cls_tokens=1,
            pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True), thermal_stem=thermal_stem,
        )
        imu_stem = PatchEmbedGeneric(
            [nn.Linear(in_features=48, out_features=imu_embed_dim, bias=False)],
            norm_layer=nn.LayerNorm(normalized_shape=imu_embed_dim),
        )
        imu_preprocessor = IMUPreprocessor(
            img_size=[6, 2000], num_cls_tokens=1, kernel_size=8, embed_dim=imu_embed_dim,
            pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True), imu_stem=imu_stem,
        )
        return nn.ModuleDict({
            ModalityType.VISION: rgbt_preprocessor, ModalityType.TEXT: text_preprocessor,
            ModalityType.AUDIO: audio_preprocessor, ModalityType.DEPTH: depth_preprocessor,
            ModalityType.THERMAL: thermal_preprocessor, ModalityType.IMU: imu_preprocessor,
        })

    def _create_modality_trunks(self, vision_embed_dim=1024, vision_num_blocks=24, vision_num_heads=16,
                                text_embed_dim=768, text_num_blocks=12, text_num_heads=12,
                                audio_embed_dim=768, audio_num_blocks=12, audio_num_heads=12, audio_drop_path=0.0,
                                depth_embed_dim=768, depth_num_blocks=12, depth_num_heads=12, depth_drop_path=0.0,
                                thermal_embed_dim=768, thermal_num_blocks=12, thermal_num_heads=12, thermal_drop_path=0.0,
                                imu_embed_dim=512, imu_num_blocks=6, imu_num_heads=8, imu_drop_path=0.7):
        def instantiate_trunk(embed_dim, num_blocks, num_heads, pre_transformer_ln, add_bias_kv, drop_path):
            return SimpleTransformer(
                embed_dim=embed_dim, num_blocks=num_blocks, ffn_dropout_rate=0.0, drop_path_rate=drop_path,
                attn_target=partial(MultiheadAttention, embed_dim=embed_dim, num_heads=num_heads, bias=True, add_bias_kv=add_bias_kv),
                pre_transformer_layer=nn.Sequential(nn.LayerNorm(embed_dim, eps=1e-6) if pre_transformer_ln else nn.Identity(), EinOpsRearrange("b l d -> l b d")),
                post_transformer_layer=EinOpsRearrange("l b d -> b l d"),
            )
        return nn.ModuleDict({
            ModalityType.VISION: instantiate_trunk(vision_embed_dim, vision_num_blocks, vision_num_heads, True, False, 0.0),
            ModalityType.TEXT: instantiate_trunk(text_embed_dim, text_num_blocks, text_num_heads, False, False, 0.0),
            ModalityType.AUDIO: instantiate_trunk(audio_embed_dim, audio_num_blocks, audio_num_heads, False, True, audio_drop_path),
            ModalityType.DEPTH: instantiate_trunk(depth_embed_dim, depth_num_blocks, depth_num_heads, False, True, depth_drop_path),
            ModalityType.THERMAL: instantiate_trunk(thermal_embed_dim, thermal_num_blocks, thermal_num_heads, False, True, thermal_drop_path),
            ModalityType.IMU: instantiate_trunk(imu_embed_dim, imu_num_blocks, imu_num_heads, False, True, imu_drop_path),
        })

    def _create_modality_heads(self, out_embed_dim, vision_embed_dim, text_embed_dim, audio_embed_dim,
                               depth_embed_dim, thermal_embed_dim, imu_embed_dim):
        return nn.ModuleDict({
            ModalityType.VISION: nn.Sequential(nn.LayerNorm(normalized_shape=vision_embed_dim, eps=1e-6), SelectElement(index=0), nn.Linear(vision_embed_dim, out_embed_dim, bias=False)),
            ModalityType.TEXT: SelectEOSAndProject(proj=nn.Sequential(nn.LayerNorm(normalized_shape=text_embed_dim, eps=1e-6), nn.Linear(text_embed_dim, out_embed_dim, bias=False))),
            ModalityType.AUDIO: nn.Sequential(nn.LayerNorm(normalized_shape=audio_embed_dim, eps=1e-6), SelectElement(index=0), nn.Linear(audio_embed_dim, out_embed_dim, bias=False)),
            ModalityType.DEPTH: nn.Sequential(nn.LayerNorm(normalized_shape=depth_embed_dim, eps=1e-6), SelectElement(index=0), nn.Linear(depth_embed_dim, out_embed_dim, bias=False)),
            ModalityType.THERMAL: nn.Sequential(nn.LayerNorm(normalized_shape=thermal_embed_dim, eps=1e-6), SelectElement(index=0), nn.Linear(thermal_embed_dim, out_embed_dim, bias=False)),
            ModalityType.IMU: nn.Sequential(nn.LayerNorm(normalized_shape=imu_embed_dim, eps=1e-6), SelectElement(index=0), nn.Dropout(p=0.5), nn.Linear(imu_embed_dim, out_embed_dim, bias=False)),
        })

    def _create_modality_postprocessors(self, out_embed_dim):
        return nn.ModuleDict({
            ModalityType.VISION: Normalize(dim=-1),
            ModalityType.TEXT: nn.Sequential(Normalize(dim=-1), LearnableLogitScaling(learnable=True)),
            ModalityType.AUDIO: nn.Sequential(Normalize(dim=-1), LearnableLogitScaling(logit_scale_init=20.0, learnable=False)),
            ModalityType.DEPTH: nn.Sequential(Normalize(dim=-1), LearnableLogitScaling(logit_scale_init=5.0, learnable=False)),
            ModalityType.THERMAL: nn.Sequential(Normalize(dim=-1), LearnableLogitScaling(logit_scale_init=10.0, learnable=False)),
            ModalityType.IMU: nn.Sequential(Normalize(dim=-1), LearnableLogitScaling(logit_scale_init=5.0, learnable=False)),
        })

    def spatial_attention(self, audio_tokens, vision_tokens, layer_a2v, layer_v2a):
        def process_sattn(a_cls_token, v_patch_tokens, layer_a2v):
            a_cls_token = layer_a2v(a_cls_token) 
            norm_a_cls_token = F.normalize(a_cls_token, dim=-1)
            norm_v_patch_tokens = F.normalize(v_patch_tokens, dim=-1) 
            av_simm = torch.sum(torch.mul(norm_a_cls_token, norm_v_patch_tokens), dim=-1)
            updated_v_patch_tokens = v_patch_tokens + torch.mul(v_patch_tokens, av_simm.unsqueeze(-1))
            return updated_v_patch_tokens

        a_cls_token = audio_tokens[0, :, :].unsqueeze(0)
        a_patch_tokens = audio_tokens[1:, :, :] 
        v_cls_token = vision_tokens[0, :, :].unsqueeze(0)
        v_patch_tokens = vision_tokens[1:, :, :]

        updated_a_patch_tokens = process_sattn(v_cls_token, a_patch_tokens, layer_v2a)
        updated_v_patch_tokens = process_sattn(a_cls_token, v_patch_tokens, layer_a2v)

        updated_a_tokens = torch.cat([a_cls_token, updated_a_patch_tokens], dim=0)
        updated_v_tokens = torch.cat([v_cls_token, updated_v_patch_tokens], dim=0)
        return updated_a_tokens, updated_v_tokens

    def _ablate_centers(self, centers, modality):
        mode = getattr(self, f'semantic_ablation_{modality}')
        scale = getattr(self, f'semantic_noise_scale_{modality}')
        if mode == 'none':
            return centers
        B, T, D = centers.shape
        if mode == 'shuffle':
            out = centers.clone()
            for b in range(B):
                mask = centers[b].abs().sum(dim=-1) > 1e-8
                if mask.sum() > 1:
                    perm = torch.randperm(int(mask.sum()), device=centers.device)
                    out[b, mask] = out[b, mask][perm]
            return out
        elif mode == 'zero':
            return torch.zeros_like(centers)
        elif mode == 'noise':
            mask = (centers.abs().sum(dim=-1, keepdim=True) > 1e-8).float()
            return centers + torch.randn_like(centers) * scale * mask
        return centers

    def forward(self, inputs):
        outputs = {}
        inputs_temp = {}
        reduce_flag = {}
        
        for modality_key, modality_value in inputs.items():
            if modality_key == 'raw_guide_centers' or modality_key == 'raw_audio_guide_centers': continue
            reduce_list = (modality_value.ndim >= 5)
            if reduce_list:
                B, S = modality_value.shape[:2]
                modality_value = modality_value.reshape(B * S, *modality_value.shape[2:])
                reduce_flag[modality_key] = True
            else:
                reduce_flag[modality_key] = False
            
            if modality_value is not None:
                modality_value = self.modality_preprocessors[modality_key](**{modality_key: modality_value})
                inputs_temp[modality_key] = modality_value

## --- TEXT ---
        text_modality_key = 'text'
        text_trunk_inputs = inputs_temp[text_modality_key]['trunk']
        text_head_inputs = inputs_temp[text_modality_key]['head']
        text_transformer_blocks = self.modality_trunks[text_modality_key].blocks
        text_tokens = text_trunk_inputs['tokens']
        
        if self.modality_trunks[text_modality_key].pre_transformer_layer:
            text_tokens = self.modality_trunks[text_modality_key].pre_transformer_layer(text_tokens)
        
        outputs['raw_text_input'] = text_tokens.detach().clone() 
        outputs['aux_loss'] = torch.tensor(0.0, device=text_tokens.device)
        outputs['prompts'] = None 

        if self.manifold_align_flag:
            manifold_prompts = self.manifold_adapter(text_tokens)
            prompts_batch = manifold_prompts.permute(1, 0, 2) 
            prompts_norm = F.normalize(prompts_batch, p=2, dim=-1)
            gram_batch = torch.bmm(prompts_norm, prompts_norm.transpose(1, 2))
            N = prompts_batch.shape[1]
            B_size = prompts_batch.shape[0]
            eye = torch.eye(N, device=prompts_batch.device).unsqueeze(0).expand(B_size, -1, -1)
            diversity_loss = ((gram_batch - eye) ** 2).sum() / (B_size * N * (N - 1))
            outputs['aux_loss'] = diversity_loss

            outputs['prompts'] = manifold_prompts.permute(1, 0, 2).detach().clone() 
            if 'seq_len' in text_head_inputs:
                text_head_inputs['seq_len'] = text_head_inputs['seq_len'] + manifold_prompts.shape[0]
            text_tokens = torch.cat([manifold_prompts, text_tokens], dim=0)

            original_mask = text_trunk_inputs['attn_mask']
            if original_mask is not None:
                N = manifold_prompts.shape[0]
                L = original_mask.shape[0]
                new_mask = torch.zeros((L + N, L + N), device=original_mask.device, dtype=original_mask.dtype)
                new_mask[N:, N:] = original_mask
                new_mask[:N, N:] = float("-inf")
                text_trunk_inputs['attn_mask'] = new_mask

        for blk in text_transformer_blocks:
            text_tokens = blk(text_tokens, attn_mask=text_trunk_inputs['attn_mask'])
        
        if self.modality_trunks[text_modality_key].post_transformer_layer:
            text_tokens = self.modality_trunks[text_modality_key].post_transformer_layer(text_tokens)

        if self.manifold_align_flag:
            N = self.manifold_token_num
            encoded_prompts = text_tokens[:N, :, :]
            encoded_prompts = encoded_prompts.permute(1, 0, 2)
            text_projector = self.modality_heads[text_modality_key].proj
            encoded_prompts = text_projector(encoded_prompts) 
            encoded_prompts = F.normalize(encoded_prompts, dim=-1)
            outputs['prompts_proj'] = encoded_prompts 
            
        text_modality_value = text_tokens
        text_modality_value = self.modality_heads[text_modality_key](text_modality_value, **text_head_inputs)
        text_modality_value = self.modality_postprocessors[text_modality_key](text_modality_value)
        if reduce_flag[text_modality_key]:
            text_modality_value = text_modality_value.reshape(B, S, -1)
            
        if self.text_tune_flag:
            text_modality_value = text_modality_value + torch.sigmoid(self.task_res_alpha) * self.task_res_text_learner(text_modality_value)

        if ModalityType.AUDIO not in inputs_temp or ModalityType.VISION not in inputs_temp:
            outputs[text_modality_key] = text_modality_value
            outputs[ModalityType.AUDIO] = None
            outputs[ModalityType.VISION] = None
            outputs['pred'] = None

            return outputs

## --- AUDIO & VISION ---
        audio_modality_key = 'audio'
        audio_trunk_inputs = inputs_temp[audio_modality_key]['trunk']
        audio_head_inputs = inputs_temp[audio_modality_key]['head']
        audio_transformer_blocks = self.modality_trunks[audio_modality_key].blocks
        audio_tokens = audio_trunk_inputs['tokens'] 
        if self.modality_trunks[audio_modality_key].pre_transformer_layer:
            audio_tokens = self.modality_trunks[audio_modality_key].pre_transformer_layer(audio_tokens)

        vision_modality_key = 'vision'
        vision_trunk_inputs = inputs_temp[vision_modality_key]['trunk']
        vision_head_inputs = inputs_temp[vision_modality_key]['head']
        vison_transformer_blocks = self.modality_trunks[vision_modality_key].blocks
        vision_tokens = vision_trunk_inputs['tokens']
        if self.modality_trunks[vision_modality_key].pre_transformer_layer:
            vision_tokens = self.modality_trunks[vision_modality_key].pre_transformer_layer(vision_tokens)

        sattn_type = self.sattn_flag
        if sattn_type == 'none':
            for audio_blk in audio_transformer_blocks: audio_tokens = audio_blk(audio_tokens, attn_mask=None)
            for vision_blk in vison_transformer_blocks: vision_tokens = vision_blk(vision_tokens, attn_mask=None)
        else:
            a2v_modulelist, v2a_modulelist = self.spatial_av_layers[audio_modality_key], self.spatial_av_layers[vision_modality_key]
            a2v_layer_ids, v2a_layer_ids = self.spatial_av_attn_layer_ids
            audio_blocks_num, vision_blocks_num = len(audio_transformer_blocks), len(vison_transformer_blocks)
            a_blk_id, v_blk_id = 0, 0
            for i in range(len(a2v_layer_ids)):
                while(a_blk_id <= a2v_layer_ids[i]):
                    audio_tokens = audio_transformer_blocks[a_blk_id](audio_tokens, attn_mask=None)
                    a_blk_id += 1
                while(v_blk_id <= v2a_layer_ids[i]):
                    vision_tokens = vison_transformer_blocks[v_blk_id](vision_tokens, attn_mask=None)
                    v_blk_id += 1
                audio_tokens, vision_tokens = self.spatial_attention(audio_tokens, vision_tokens, a2v_modulelist[i], v2a_modulelist[i])
            for ai in range(a_blk_id, audio_blocks_num): audio_tokens = audio_transformer_blocks[ai](audio_tokens, attn_mask=None)
            for vi in range(v_blk_id, vision_blocks_num): vision_tokens = vison_transformer_blocks[vi](vision_tokens, attn_mask=None)

        if self.modality_trunks[audio_modality_key].post_transformer_layer:
            audio_tokens = self.modality_trunks[audio_modality_key].post_transformer_layer(audio_tokens)
        audio_modality_value = audio_tokens
        audio_modality_value = self.modality_heads[audio_modality_key](audio_modality_value, **audio_head_inputs)
        audio_modality_value = self.modality_postprocessors[audio_modality_key](audio_modality_value)
        if reduce_flag[audio_modality_key]:
            audio_modality_value = audio_modality_value.reshape(B, S, -1)

        if self.modality_trunks[vision_modality_key].post_transformer_layer:
            vision_tokens = self.modality_trunks[vision_modality_key].post_transformer_layer(vision_tokens)
        vision_modality_value = vision_tokens
        vision_modality_value = self.modality_heads[vision_modality_key](vision_modality_value, **vision_head_inputs)
        vision_modality_value = self.modality_postprocessors[vision_modality_key](vision_modality_value)
        if reduce_flag[vision_modality_key]:
            vision_modality_value = vision_modality_value.reshape(B, S, -1)
        

## 1. ImageBind
        vision_ib_feat = vision_modality_value
        audio_ib_feat = audio_modality_value
        
## CenterLoss ImageBind
        outputs['vision_raw'] = F.normalize(vision_ib_feat, p=2, dim=-1)
        outputs['video_features_for_cl'] = outputs['vision_raw']
        outputs['audio_raw'] = F.normalize(audio_ib_feat, p=2, dim=-1)
        outputs['audio_features_for_cl'] = outputs['audio_raw']
## 2. Student ImageBind -> ()
        if self.tattn_flag:
            audio_student, vision_student = self.temporal_av_layer(audio_ib_feat, vision_ib_feat) 
        else:
            audio_student, vision_student = audio_ib_feat, vision_ib_feat
## Student
        vision_final_feat = vision_student
        audio_final_feat = audio_student
## 3. Teacher ImageBind -> -> ()
        if self.use_cluster_module and self.training:
## Student Loss (Student Teacher)
            outputs['vision_student'] = vision_student
            outputs['audio_student'] = audio_student
            
## --- Vision Teacher ---
            if ('raw_guide_centers' in inputs) and (inputs['raw_guide_centers'] is not None):
                raw_guide_centers = inputs['raw_guide_centers']
                projected_centers = self.cluster_proj(raw_guide_centers) 
                projected_centers = self._ablate_centers(projected_centers, 'v')
                
                q = self.norm_video(vision_student)
                attn_out, _ = self.cross_attn(query=q, key=projected_centers, value=projected_centers, need_weights=False)
                
## Teacher = ImageBind +
                vision_final_feat = vision_student + self.cross_attn_dropout(attn_out)
                outputs['projected_centers'] = projected_centers
                
## --- Audio Teacher ---
            if ('raw_audio_guide_centers' in inputs) and (inputs['raw_audio_guide_centers'] is not None):
                raw_audio_guide_centers = inputs['raw_audio_guide_centers']
                projected_audio_centers = self.audio_cluster_proj(raw_audio_guide_centers) 
                projected_audio_centers = self._ablate_centers(projected_audio_centers, 'a')
                
                q_a = self.norm_audio(audio_student)
                attn_out_a, _ = self.audio_cross_attn(query=q_a, key=projected_audio_centers, value=projected_audio_centers, need_weights=False)
                
## Teacher = ImageBind +
                audio_final_feat = audio_student + self.audio_cross_attn_dropout(attn_out_a)
                outputs['projected_audio_centers'] = projected_audio_centers
## Teacher Student
        outputs[text_modality_key] = text_modality_value
        outputs[audio_modality_key] = audio_final_feat
        outputs[vision_modality_key] = vision_final_feat
        outputs['pred'] = None
        outputs['alpha_v'], outputs['beta_v'] = self.alpha_v, self.beta_v
        outputs['alpha_a'], outputs['beta_a'] = self.alpha_a, self.beta_a
        
        return outputs

def imagebind_huge(pretrained=False, spatial_av_attn_layer_ids=([0], [0]), sattn_flag='none',
                   tattn_flag=False, sa_layer_num=1, xa_layer_num=1, feat_dim=1024, hid_dim=256,
                   d_ff=512, head_num=1, dropout=0.1, use_adj_in_attn=False, gamma=0.6, bias=0.2,
                   use_mask_in_attn=False, win_size=4, norm_flag=None, text_tune_flag=False,
                   manifold_align_flag=True, manifold_token_num=4, 
                    use_cluster_module=False,
                    semantic_ablation_v='none',
                    semantic_ablation_a='none',
                    semantic_noise_scale_v=1.0,
                    semantic_noise_scale_a=1.0):
    model = ImageBindModel(
        vision_embed_dim=1280, vision_num_blocks=32, vision_num_heads=16,
        text_embed_dim=1024, text_num_blocks=24, text_num_heads=16,
        out_embed_dim=1024, audio_drop_path=0.1, imu_drop_path=0.7,
        spatial_av_attn_layer_ids=spatial_av_attn_layer_ids, sattn_flag=sattn_flag,
        tattn_flag=tattn_flag, sa_layer_num=sa_layer_num, xa_layer_num=xa_layer_num,
        feat_dim=feat_dim, hid_dim=hid_dim, d_ff=d_ff, head_num=head_num, dropout=dropout,
        use_adj_in_attn=use_adj_in_attn, gamma=gamma, bias=bias, use_mask_in_attn=use_mask_in_attn,
        win_size=win_size, norm_flag=norm_flag, text_tune_flag=text_tune_flag,
        manifold_align_flag=manifold_align_flag, manifold_token_num=manifold_token_num,
        use_cluster_module=use_cluster_module,
        semantic_ablation_v=semantic_ablation_v,
        semantic_ablation_a=semantic_ablation_a,
        semantic_noise_scale_v=semantic_noise_scale_v,
        semantic_noise_scale_a=semantic_noise_scale_a,
    )

    def initialize_imagebind_weights(model):
        imagebind_model_dict = model.state_dict()
        pretrained_path = '/home/liuchi/OV-AVEL/checkpoint/imagebind_huge.pth'
        if not os.path.exists(pretrained_path): pretrained_path = ".checkpoints/imagebind_huge.pth"
        
        if os.path.exists(pretrained_path):
            pretrained_state_dicts = torch.load(pretrained_path)
            state_dict = {k : v for k, v in pretrained_state_dicts.items() if k in imagebind_model_dict.keys()}
            imagebind_model_dict.update(state_dict)
            print(f"==> Load pretrained Imagemodel parameters from {pretrained_path}")
            model.load_state_dict(imagebind_model_dict)
        else:
             print("==> Pretrained path not found, skipping loading.")
        return model

    if pretrained: model = initialize_imagebind_weights(model)
    return model
