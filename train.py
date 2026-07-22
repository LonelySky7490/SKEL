import os
import time
import random
import json
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from sklearn.cluster import KMeans

from utils.visualizer import check_space_distribution
import numpy as np
from configs.opts import parser
from utils import AverageMeter, Prepare_logger, get_and_save_args
from utils.eval_metrics import segment_level, event_level

from imagebind_finetune.models import imagebind_model as imagebind_model_fully
from imagebind_finetune.models.imagebind_model import ModalityType
from einops import rearrange, repeat, reduce
from dataloader import OVAVE_Dataset
from config import cfg
import pdb

parser.add_argument('--num_clusters', type=int, default=20, help='Number of clusters for KMeans')
parser.add_argument('--center_dropout', type=float, default=0.5, help='Probability to drop GT center during training')

parser.add_argument('--disable_cluster_module', action='store_true',
                    help='Use for ablation study: completely remove cluster params')

parser.add_argument('--weight_cl', type=float, default=0.1,
                    help='Weight for cluster center loss (loss_cl)')
parser.add_argument('--weight_aux', type=float, default=100.0,
                    help='Weight for auxiliary loss (aux_loss)')
parser.add_argument('--weight_guide', type=float, default=0.5,
                    help='Weight for guide loss (guide_loss)')
parser.add_argument('--weight_kd', type=float, default=0.5,
                    help='Weight for knowledge distillation loss (loss_kd)')

parser.add_argument('--result_file', type=str, default='',
                    help='If set, write best metric to this file for Optuna to read')
parser.add_argument('--manifold_token_num', type=int, default=4,
                    help='Number of manifold tokens for Semantic Manifold Alignment')

args = get_and_save_args(parser)

SEED = args.seed
random.seed(SEED)
np.random.seed(seed=SEED)
torch.manual_seed(seed=SEED)
torch.cuda.manual_seed_all(seed=SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if not os.path.exists(args.snapshot_pref):
    os.makedirs(args.snapshot_pref, exist_ok=True)
if os.path.isfile(args.resume):
    args.snapshot_pref = os.path.dirname(args.resume)

logger = Prepare_logger(args, eval=args.evaluate)

if cfg.SA.SA_ATTN_FLAG:
    if cfg.SA.SA_ATTN_TYPE == 'bothFirst':
        audio_layer_ids = list(range(cfg.SA.SA_BOTH_LAYER_K))
        visual_layer_ids = list(range(cfg.SA.SA_BOTH_LAYER_K))
        sattn_type = 'bothFirst'
    elif cfg.SA.SA_ATTN_TYPE == 'bothLast':
        audio_layer_ids = list(range(12 - cfg.SA.SA_BOTH_LAYER_K, 12))
        visual_layer_ids = list(range(32 - cfg.SA.SA_BOTH_LAYER_K, 32))
        sattn_type = 'bothLast'
    elif cfg.SA.SA_ATTN_TYPE == 'evenFirst':
        audio_layer_ids = list(range(0, 12, 3)) + [11]
        visual_layer_ids = list(range(0, 32, 8)) + [31]
        audio_layer_ids = audio_layer_ids[:cfg.SA.SA_EVEN_LAYER_K]
        visual_layer_ids = visual_layer_ids[:cfg.SA.SA_EVEN_LAYER_K]
        sattn_type = 'evenFirst'
    elif cfg.SA.SA_ATTN_TYPE == 'evenLast':
        audio_layer_ids = list(range(0, 12, 3)) + [11]
        visual_layer_ids = list(range(0, 32, 8)) + [31]
        audio_layer_ids = audio_layer_ids[-cfg.SA.SA_EVEN_LAYER_K:]
        visual_layer_ids = visual_layer_ids[-cfg.SA.SA_EVEN_LAYER_K:]
        sattn_type = 'evenLast'
    elif cfg.SA.SA_ATTN_TYPE == 'fixedBlkids':
        audio_layer_ids = cfg.SA.SA_FIXED_LAYER_A
        visual_layer_ids = cfg.SA.SA_FIXED_LAYER_V
        sattn_type = 'fixedBlkids'
    else:
        raise NotImplementedError
else:
    audio_layer_ids = []
    visual_layer_ids = []
    sattn_type = 'none'

logger.info(f"==> Spatial Attention Mode: {sattn_type}")
logger.info(f"==> Spatial Attention audio_layer_ids: {audio_layer_ids}")
logger.info(f"==> Spatial Attention visual_layer_ids: {visual_layer_ids}")

if not args.evaluate:
    logger.info(f'\nCreating folder: {args.snapshot_pref}')
    logger.info('\nRuntime args\n\n{}\n'.format(json.dumps(vars(args), indent=4)))
else:
    logger.info(f'\nLog file will be save in {args.snapshot_pref}/Eval_{args.test_data_type}.log.')
    logger.info('\nRuntime args\n\n{}\n'.format(json.dumps(vars(args), indent=4)))


class CenterLoss(nn.Module):
    def __init__(self):
        super(CenterLoss, self).__init__()

    def forward(self, features, centers):
        diff = features - centers
        loss_part = diff.pow(2).sum(dim=1)
        loss = 0.5 * loss_part.mean()
        return loss


def run_text_clustering(model, train_dataloader, n_clusters=20):
    print(f"\n[Clustering] Running Text Clustering (K={n_clusters})...")
    class_names = None
    dataset = train_dataloader.dataset
    if hasattr(dataset, 'classes'):
        class_names = dataset.classes
    elif hasattr(dataset, 'label_map'):
        label_map = dataset.label_map
        if isinstance(label_map, dict):
            class_names = [label_map[i] for i in range(len(label_map))]
        else:
            class_names = label_map
    elif hasattr(dataset, 'labels'):
        class_names = dataset.labels
    if class_names is None:
        print("[Warning] Could not find class names in dataset. Will show IDs only.")

    model.eval()
    text_feats_norm = None
    for _, _, text, _, _, _, _ in train_dataloader:
        text_inputs = text[0].cuda()
        inputs = {ModalityType.TEXT: text_inputs}
        with torch.no_grad():
            outputs = model(inputs)
            text_feats_norm = outputs['text']
        break

    if text_feats_norm is None: return None, None

    feats_np = text_feats_norm.cpu().numpy()
    real_k = min(n_clusters, len(feats_np))
    kmeans = KMeans(n_clusters=real_k, init='k-means++', n_init=10, random_state=42)
    labels = kmeans.fit_predict(feats_np)

    unique_labels = sorted(list(set(labels)))
    logger.info(f"  > Found {len(unique_labels)} clusters.")

    current_cluster_bank = {}
    current_index_to_cluster = {}

    logger.info(f"\n{'-' * 30} Cluster Semantic Report {'-' * 30}")
    with torch.no_grad():
        for lbl in unique_labels:
            indices = np.where(labels == lbl)[0]
            indices_list = indices.tolist()
            content_str = ""
            if class_names is not None:
                names = [class_names[i] for i in indices_list if i < len(class_names)]
                content_str = f"{names}"
            else:
                content_str = f"IDs: {indices_list}"
            logger.info(f"  [Cluster {lbl:02d}] ({len(indices)} events): {content_str}")
            raw_group_feats = text_feats_norm[indices]
            raw_center = torch.mean(raw_group_feats, dim=0)
            current_cluster_bank[lbl] = raw_center.cpu()
            for original_idx in indices:
                current_index_to_cluster[int(original_idx)] = lbl
    logger.info(f"{'-' * 85}\n")
    return current_cluster_bank, current_index_to_cluster


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    train_dataloader = DataLoader(OVAVE_Dataset(split='train', debug=args.debug), batch_size=args.train_batch_size,
                                  shuffle=True, num_workers=8, pin_memory=True)
    val_dataloader = DataLoader(OVAVE_Dataset(split='val', test_data_type=args.val_data_type, debug=args.debug),
                                batch_size=args.val_batch_size, shuffle=False, num_workers=8, pin_memory=True)
    test_dataloader = DataLoader(OVAVE_Dataset(split='test', test_data_type=args.test_data_type, debug=args.debug),
                                 batch_size=args.test_batch_size, shuffle=False, num_workers=8, pin_memory=True)

    mainModel = imagebind_model_fully.imagebind_huge(
        pretrained=True,
        spatial_av_attn_layer_ids=(audio_layer_ids, visual_layer_ids),
        sattn_flag=sattn_type,
        tattn_flag=cfg.TA.TA_ATTN_FLAG,
        sa_layer_num=cfg.TA.SA_LAYER_NUM,
        xa_layer_num=cfg.TA.XA_LAYER_NUM,
        feat_dim=1024,
        hid_dim=cfg.TA.HIDDEN_DIM,
        d_ff=cfg.TA.FF_DIM,
        head_num=cfg.TA.HEAD_NUM,
        dropout=cfg.TA.DROPOUT,
        use_adj_in_attn=cfg.TA.USE_ADJ_IN_ATTN,
        gamma=cfg.TA.GAMMA,
        bias=cfg.TA.BIAS,
        use_mask_in_attn=cfg.TA.USE_MASK_IN_ATTN,
        win_size=cfg.TA.WIN_SIZE,
        norm_flag=cfg.TA.NORM_FLAG,
        text_tune_flag=cfg.TEXT_TUNE_FLAG,
        manifold_align_flag=cfg.manifold_align_flag,
        manifold_token_num=args.manifold_token_num,
        use_cluster_module=not args.disable_cluster_module,
    )
    mainModel.to(device)

    for name, params in mainModel.named_parameters():
        if 'spatial_av_layers' in name:
            params.requires_grad = True
        elif 'temporal_av_layer' in name:
            params.requires_grad = True
        elif 'task_res' in name:
            params.requires_grad = True
        elif 'manifold_adapter' in name:
            params.requires_grad = True
        elif 'cluster_proj' in name or 'cross_attn' in name or 'norm_video' in name \
                or 'audio_cluster_proj' in name or 'audio_cross_attn' in name or 'norm_audio' in name:
            print(f"Unfreezing Clustering Layer: {name}")
            params.requires_grad = True
        elif 'alpha_' in name or 'beta_' in name:
            print(f"Unfreezing Calibration Params: {name}")
            params.requires_grad = False
        else:
            params.requires_grad = False

    for name, params in mainModel.named_parameters():
        if params.requires_grad: print(name)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, mainModel.parameters()), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.75)

    criterion = nn.BCEWithLogitsLoss().cuda()
    criterion_event = nn.CrossEntropyLoss().cuda()
    criterion_center = CenterLoss().cuda()

    if os.path.isfile(args.resume):
        logger.info(f"\nLoading Checkpoint: {args.resume}\n")
        mainModel.load_state_dict(torch.load(args.resume))
    elif args.resume != "" and (not os.path.isfile(args.resume)):
        logger.info(f"\nCheckpoint {args.resume} not found. Starting from scratch.\n")
        raise FileNotFoundError
    else:
        logger.info(f"\nNo checkpoint specified. Starting from scratch.\n")

    latest_cluster_bank = None
    latest_index_to_cluster = None

    if args.evaluate:
        logger.info(f"\nStart testing..")
        if args.disable_cluster_module:
            latest_cluster_bank = None
            latest_index_to_cluster = None
        else:
            latest_cluster_bank, latest_index_to_cluster = run_text_clustering(mainModel, train_dataloader,
                                                                               n_clusters=args.num_clusters)
        validate_epoch(mainModel, test_dataloader, criterion, criterion_event, epoch=0, eval_only=True)

        return

    best_accuracy = 0
    for epoch in range(args.n_epoch):
        loss = train_epoch(mainModel, train_dataloader, criterion, criterion_event, criterion_center,
                           optimizer, epoch, latest_cluster_bank, latest_index_to_cluster)

        if ((epoch) % args.eval_freq == 0) or (epoch == args.n_epoch - 1):
            acc = validate_epoch(mainModel, test_dataloader, criterion, criterion_event, epoch)
            if acc > best_accuracy:
                best_accuracy = acc
                logger.info(f'best average result at epoch-{epoch}: {best_accuracy:.4f}')
                save_checkpoint(mainModel.state_dict(), top1=best_accuracy, task='FullySupervised', epoch=epoch + 1,
                                seed=SEED)
            elif epoch < 2:

                logger.info(f'saving checkpoint for epoch-{epoch} (first 2 epochs, acc: {acc:.4f})')
                save_checkpoint(mainModel.state_dict(), top1=acc, task='FullySupervised', epoch=epoch + 1,
                                seed=SEED)

        if args.disable_cluster_module:
            latest_cluster_bank = None
            latest_index_to_cluster = None
        else:
            latest_cluster_bank, latest_index_to_cluster = run_text_clustering(mainModel, train_dataloader,
                                                                               n_clusters=args.num_clusters)

        scheduler.step()

    if args.result_file:
        os.makedirs(os.path.dirname(args.result_file), exist_ok=True)
        with open(args.result_file, 'w') as f:
            json.dump({'best_metric': best_accuracy}, f)
        logger.info(f"[Optuna] Best metric {best_accuracy:.6f} saved to {args.result_file}")

    return best_accuracy


def train_epoch(model, train_dataloader, criterion, criterion_event, criterion_center, optimizer, epoch,
                cluster_bank=None, index_to_cluster=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    train_acc = AverageMeter()
    train_seg_fscore = AverageMeter()
    train_eve_fscore = AverageMeter()
    end_time = time.time()

    model.train()


    has_bank = (cluster_bank is not None)
    if has_bank: print(f"  [Train] Injecting Centers with Dropout {args.center_dropout}...")

    for n_iter, batch_data in enumerate(train_dataloader):
        optimizer.zero_grad(set_to_none=True)
        audio, visual, text, full_label, avc_label, category_label, vid_name = batch_data
        audio_inputs = audio.squeeze(1).cuda()
        visual_inputs = visual.cuda()
        text_inputs = text[0].cuda()
        avc_labels = avc_label.cuda()
        full_labels = full_label.cuda()
        category_labels = category_label.reshape(-1).cuda().long()

        bs = visual_inputs.size(0)
        seq_len = visual_inputs.size(1)
        feat_dim = 1024

        guide_centers_v = None
        guide_centers_a = None
        gt_centers_tensor_v = None
        gt_centers_tensor_a = None
        valid_cl_mask_v = None
        valid_cl_mask_a = None

        if has_bank:
            guide_centers_v = torch.zeros(bs, seq_len, feat_dim).cuda()
            guide_centers_a = torch.zeros(bs, seq_len, feat_dim).cuda()
            bg_id = cfg.TRAIN_BG_CLASS_ID

            keep_center_mask_v = (torch.rand(bs) > args.center_dropout).cuda()
            keep_center_mask_a = (torch.rand(bs) > args.center_dropout).cuda()

            gt_centers_tensor_v = torch.zeros(bs, seq_len, feat_dim).cuda()
            gt_centers_tensor_a = torch.zeros(bs, seq_len, feat_dim).cuda()
            valid_cl_mask_v = torch.zeros(bs, seq_len).cuda()
            valid_cl_mask_a = torch.zeros(bs, seq_len).cuda()

            for b in range(bs):
                should_keep_v = keep_center_mask_v[b].item()
                should_keep_a = keep_center_mask_a[b].item()
                for t in range(seq_len):
                    lbl = full_labels[b, t].item()
                    if lbl != bg_id:
                        if (lbl in index_to_cluster) and (index_to_cluster[lbl] in cluster_bank):
                            cid = index_to_cluster[lbl]
                            c_raw = cluster_bank[cid].cuda()

                            if should_keep_v: guide_centers_v[b, t] = c_raw
                            gt_centers_tensor_v[b, t] = c_raw
                            valid_cl_mask_v[b, t] = 1.0

                            if should_keep_a: guide_centers_a[b, t] = c_raw
                            gt_centers_tensor_a[b, t] = c_raw
                            valid_cl_mask_a[b, t] = 1.0

        inputs = {
            ModalityType.TEXT: text_inputs.cuda(),
            ModalityType.VISION: visual_inputs.cuda(),
            ModalityType.AUDIO: audio_inputs.cuda(),
            'raw_guide_centers': guide_centers_v,
            'raw_audio_guide_centers': guide_centers_a
        }

        embeddings = model(inputs)

        audio_feas = embeddings['audio']
        visual_feas = embeddings['vision']
        text_feas = embeddings['text']
        text_feas = text_feas.unsqueeze(0).repeat(bs, 1, 1)

        alpha_v, beta_v = embeddings['alpha_v'], embeddings['beta_v']
        alpha_a, beta_a = embeddings['alpha_a'], embeddings['beta_a']

        simm_at = compute_cross_modal_similarity(audio_feas, text_feas)
        simm_vt = compute_cross_modal_similarity(visual_feas, text_feas)

        margin_v = simm_vt.max(dim=-1)[0] - simm_vt.mean(dim=-1) # [B, T]
        margin_a = simm_at.max(dim=-1)[0] - simm_at.mean(dim=-1) # [B, T]

        Q_v_raw = margin_v.max(dim=-1, keepdim=True)[0].unsqueeze(-1) # [B, 1, 1]
        Q_a_raw = margin_a.max(dim=-1, keepdim=True)[0].unsqueeze(-1) # [B, 1, 1]

        Q_v_calib = Q_v_raw * alpha_v + beta_v
        Q_a_calib = Q_a_raw * alpha_a + beta_a

        Q_v = F.softplus(Q_v_calib)
        Q_a = F.softplus(Q_a_calib)

        w_v = Q_v / (Q_v + Q_a + 1e-8) # [B, 1, 1]
        w_a = Q_a / (Q_v + Q_a + 1e-8) # [B, 1, 1]


        drop_thresh = 0.1
        valid_mask = ((w_v >= drop_thresh) & (w_a >= drop_thresh)).float() # [B, 1, 1]

        sim_v_pos = F.relu(simm_vt)
        sim_a_pos = F.relu(simm_at)

        pred_logits = torch.sqrt(F.relu(simm_at * simm_vt))


        pred_logits = pred_logits * valid_mask

        x_min = (pred_logits.min(-1)[0]).unsqueeze(-1)
        x_max = (pred_logits.max(-1)[0]).unsqueeze(-1)
        pred_logits = (pred_logits - x_min) / (x_max - x_min + 1e-8) * valid_mask


        if torch.sum(pred_logits.isnan()) > 0: continue

        loss = criterion_event(pred_logits.permute(0, 2, 1), full_labels)

        loss = loss + args.weight_aux * embeddings.get('aux_loss', torch.tensor(0.0).cuda())

        all_class_prompts = embeddings.get('prompts_proj', None)
        if all_class_prompts is None: all_class_prompts = embeddings.get('prompts', None)

        if all_class_prompts is not None:
            selected_prompts = all_class_prompts[category_labels]
            bg_id = cfg.TRAIN_BG_CLASS_ID

            fg_mask = (full_labels != bg_id).float()
            bg_mask = (full_labels == bg_id).float()

            v_feat = embeddings['vision'].detach()
            a_feat = embeddings['audio'].detach()

            fg_count = torch.clamp(fg_mask.sum(dim=1, keepdim=True), min=1e-8)
            bg_count = torch.clamp(bg_mask.sum(dim=1, keepdim=True), min=1e-8)

            v_fg_avg = (v_feat * fg_mask.unsqueeze(-1)).sum(dim=1) / fg_count
            a_fg_avg = (a_feat * fg_mask.unsqueeze(-1)).sum(dim=1) / fg_count

            v_bg_avg = (v_feat * bg_mask.unsqueeze(-1)).sum(dim=1) / bg_count
            a_bg_avg = (a_feat * bg_mask.unsqueeze(-1)).sum(dim=1) / bg_count

            p_visual = selected_prompts[:, 0, :]
            p_audio = selected_prompts[:, 1, :]

            sim_v_pos = F.cosine_similarity(p_visual, v_fg_avg, dim=-1)
            sim_v_neg = F.cosine_similarity(p_visual, v_bg_avg, dim=-1)

            sim_a_pos = F.cosine_similarity(p_audio, a_fg_avg, dim=-1)
            sim_a_neg = F.cosine_similarity(p_audio, a_bg_avg, dim=-1)

            margin = 0.5
            loss_v = F.relu(margin - sim_v_pos + sim_v_neg)
            loss_a = F.relu(margin - sim_a_pos + sim_a_neg)

            valid_contrastive_mask = (
                        (fg_mask.sum(dim=1) > 0) & (bg_mask.sum(dim=1) > 0) & (category_labels != bg_id)).float()

            loss_v = loss_v * valid_contrastive_mask
            loss_a = loss_a * valid_contrastive_mask
            num_valid = valid_contrastive_mask.sum() + 1e-8

            loss_v_pull = (1 - sim_v_pos) * valid_contrastive_mask
            loss_a_pull = (1 - sim_a_pos) * valid_contrastive_mask

            guide_loss = ((loss_v + loss_v_pull).sum() + (loss_a + loss_a_pull).sum()) / num_valid
            loss = loss + args.weight_guide * guide_loss

        loss_cl_v = torch.tensor(0.0).cuda()
        loss_cl_a = torch.tensor(0.0).cuda()

        if has_bank and (valid_cl_mask_v.sum() > 0) and ('video_features_for_cl' in embeddings):
            projector_v = model.cluster_proj if hasattr(model, 'cluster_proj') else model.module.cluster_proj
            gt_proj_centers_v = projector_v(gt_centers_tensor_v)
            gt_proj_centers_v = F.normalize(gt_proj_centers_v, dim=-1)
            vid_feats = embeddings['video_features_for_cl']
            diff_v = (vid_feats - gt_proj_centers_v).pow(2).sum(dim=-1)
            loss_cl_v = 0.5 * (diff_v * valid_cl_mask_v).sum() / (valid_cl_mask_v.sum() + 1e-8)

        if has_bank and (valid_cl_mask_a.sum() > 0) and ('audio_features_for_cl' in embeddings):
            projector_a = model.audio_cluster_proj if hasattr(model,
                                                              'audio_cluster_proj') else model.module.audio_cluster_proj
            gt_proj_centers_a = projector_a(gt_centers_tensor_a)
            gt_proj_centers_a = F.normalize(gt_proj_centers_a, dim=-1)
            aud_feats = embeddings['audio_features_for_cl']
            diff_a = (aud_feats - gt_proj_centers_a).pow(2).sum(dim=-1)
            loss_cl_a = 0.5 * (diff_a * valid_cl_mask_a).sum() / (valid_cl_mask_a.sum() + 1e-8)

        loss_cl = loss_cl_v + loss_cl_a
        loss = loss + args.weight_cl * loss_cl

        loss_kd_v = F.mse_loss(embeddings['vision_student'], embeddings['vision'].detach())
        loss_kd_a = F.mse_loss(embeddings['audio_student'], embeddings['audio'].detach())
        loss_kd = args.weight_kd * (loss_kd_v + loss_kd_a)
        loss = loss + loss_kd

        loss.backward()
        if args.clip_gradient is not None:
            total_norm = clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), args.clip_gradient)

        optimizer.step()

        if args.test_strategy_type == 'v1':
            acc = compute_accuracy_supervised_v1(pred_logits, full_labels, bg_flag=cfg.TRAIN_BG_CLASS_ID)
            train_acc.update(acc.item(), bs * 10)
            seg_f, eve_f = compute_seg_eve_fscores_v1(pred_logits, avc_labels, category_labels,
                                                      bg_flag=cfg.TRAIN_BG_CLASS_ID)
            train_seg_fscore.update(seg_f, n=1)
            train_eve_fscore.update(eve_f, n=1)

        losses.update(loss.item(), bs * 10)
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if n_iter % args.print_iter_freq == 0:
            logger.info(
                f'Train Epoch: [{epoch}][{n_iter}/{len(train_dataloader)}]\t'
                f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                f'CL_V {loss_cl_v.item():.4f} CL_A {loss_cl_a.item():.4f}\t'
                f'Prec@1 {train_acc.val:.3f} ({train_acc.avg:.3f})\t'
                f'Seg@F1 {train_seg_fscore.val:.3f} ({train_seg_fscore.avg:.3f})\t'
                f'Eve@F1 {train_eve_fscore.val:.3f} ({train_eve_fscore.avg:.3f})'
            )
    return losses.avg


def compute_cross_modal_similarity(tensor_a, tensor_t):
    B, T, D = tensor_a.shape
    _, C, _ = tensor_t.shape
    tensor_a_expanded = tensor_a.unsqueeze(2).expand(B, T, C, D)
    tensor_t_expanded = tensor_t.unsqueeze(1).expand(B, T, C, D)
    cos_sim = F.cosine_similarity(tensor_a_expanded, tensor_t_expanded, dim=-1)
    return cos_sim


@torch.no_grad()
def validate_epoch(model, val_dataloader, criterion, criterion_event, epoch, eval_only=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()
    seg_fscore = AverageMeter()
    eve_fscore = AverageMeter()
    end_time = time.time()

    model.eval()


    for n_iter, batch_data in enumerate(val_dataloader):
        audio, visual, text, full_label, avc_label, category_label, vid_name = batch_data
        audio_inputs = audio.squeeze(1).cuda()
        visual_inputs = visual.cuda()
        text_inputs = text[0].cuda()
        avc_labels = avc_label.cuda()
        category_labels = category_label.reshape(-1).cuda()
        full_labels = full_label.cuda()

        bs = visual_inputs.size(0)
        
        inputs = {
            ModalityType.TEXT: text_inputs.cuda(),
            ModalityType.VISION: visual_inputs.cuda(),
            ModalityType.AUDIO: audio_inputs.cuda(),
        }

        outputs = model(inputs)

        audio_feas = outputs['audio']
        visual_feas = outputs['vision']
        text_feas = outputs['text'].unsqueeze(0).repeat(bs, 1, 1)

        alpha_v, beta_v = outputs['alpha_v'], outputs['beta_v']
        alpha_a, beta_a = outputs['alpha_a'], outputs['beta_a']

        simm_at = compute_cross_modal_similarity(audio_feas, text_feas)
        simm_vt = compute_cross_modal_similarity(visual_feas, text_feas)

        margin_v = simm_vt.max(dim=-1)[0] - simm_vt.mean(dim=-1) # [B, T]
        margin_a = simm_at.max(dim=-1)[0] - simm_at.mean(dim=-1) # [B, T]

        Q_v_raw = margin_v.max(dim=-1, keepdim=True)[0].unsqueeze(-1) # [B, 1, 1]
        Q_a_raw = margin_a.max(dim=-1, keepdim=True)[0].unsqueeze(-1) # [B, 1, 1]

        Q_v_calib = Q_v_raw * alpha_v + beta_v
        Q_a_calib = Q_a_raw * alpha_a + beta_a

        Q_v = F.softplus(Q_v_calib)
        Q_a = F.softplus(Q_a_calib)

        w_v = Q_v / (Q_v + Q_a + 1e-8)
        w_a = Q_a / (Q_v + Q_a + 1e-8)

        drop_thresh = 0.1
        valid_mask = ((w_v >= drop_thresh) & (w_a >= drop_thresh)).float()

        sim_v_pos = F.relu(simm_vt)
        sim_a_pos = F.relu(simm_at)

        pred_logits = torch.sqrt(F.relu(simm_at * simm_vt))

        pred_logits = pred_logits * valid_mask

        x_min = (pred_logits.min(-1)[0]).unsqueeze(-1)
        x_max = (pred_logits.max(-1)[0]).unsqueeze(-1)
        pred_logits = (pred_logits - x_min) / (x_max - x_min + 1e-8) * valid_mask


        if args.test_strategy_type == 'v1':
            loss = criterion_event(pred_logits.permute(0, 2, 1), full_labels)
            acc = compute_accuracy_supervised_v1(pred_logits, full_labels, bg_flag=cfg.TOTAL_BG_CLASS_ID)
            accuracy.update(acc.item(), bs * 10)
            seg_f, eve_f = compute_seg_eve_fscores_v1(pred_logits, avc_labels, category_labels,
                                                      bg_flag=cfg.TOTAL_BG_CLASS_ID)
            seg_fscore.update(seg_f, n=1)
            eve_fscore.update(eve_f, n=1)

        batch_time.update(time.time() - end_time)
        end_time = time.time()
        losses.update(loss.item(), bs * 10)

        if n_iter % args.print_iter_freq == 0:
            logger.info(
                f'Test Epoch [{epoch}][{n_iter}/{len(val_dataloader)}]\t'
                f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                f'Prec@1 {accuracy.val:.3f} ({accuracy.avg:.3f})\t'
                f'Seg@F1 {seg_fscore.val:.3f} ({seg_fscore.avg:.3f})\t'
                f'Eve@F1 {eve_fscore.val:.3f} ({eve_fscore.avg:.3f})'
            )


    logger.info(
        f"\tEvaluation results (acc): {accuracy.avg:.4f}\t (Segment-level F1score): {seg_fscore.avg:.4f}\t (Event-level F1score): {eve_fscore.avg:.4f}"
    )
    return (accuracy.avg + seg_fscore.avg + eve_fscore.avg) / 3



def compute_accuracy_supervised_v1(pred_logits, full_labels, bg_flag):
    _, pred = pred_logits.max(-1)
    targets = full_labels
    correct = pred.eq(targets)
    correct_num = correct.sum().double()
    acc = correct_num / correct.numel()
    return acc


def compute_seg_eve_fscores_v1(pred_logits, avc_labels, category_labels, bg_flag=cfg.TOTAL_BG_CLASS_ID):
    def obtain_pred_mat(pred_logits, bg_flag):
        pred_logits = pred_logits.permute(0, 2, 1)
        pred_mat = torch.zeros_like(pred_logits).cuda()
        pred_cls_ids = pred_logits.max(dim=1)[1]
        B = pred_logits.shape[0]
        for i in range(B):
            class_id = pred_cls_ids[i]
            for j in range(len(class_id)):
                pred_mat[i][class_id[j]][j] = 1
        return pred_mat.cpu().data.numpy()

    def obtain_gt_mat(avc_labels, category_labels, bg_flag):
        B, T = avc_labels.shape
        K = bg_flag + 1
        gt_mat = torch.zeros(B, K, T).cuda()
        for i in range(B):
            class_id = category_labels[i].item()
            if class_id != bg_flag:
                gt_mat[i][class_id] = avc_labels[i]
                gt_mat[i][-1] = 1 - avc_labels[i]
            else:
                gt_mat[i][-1] = 1 - avc_labels[i]
        return gt_mat.cpu().data.numpy()

    pred = obtain_pred_mat(pred_logits, bg_flag)
    targets = obtain_gt_mat(avc_labels, category_labels, bg_flag)
    B = avc_labels.shape[0]
    seg_fscore = np.zeros(B)
    eve_fscore = np.zeros(B)
    for i in range(B):
        seg_f = segment_level(pred[i], targets[i])
        seg_fscore[i] = seg_f
        eve_f = event_level(pred[i], targets[i])
        eve_fscore[i] = eve_f
    return np.mean(seg_fscore), np.mean(eve_fscore)


def save_checkpoint(state_dict, top1, task, epoch, seed):
    import datetime
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = f'{args.snapshot_pref}/task_{task}_best_model_{timestamp}.pth.tar'
    torch.save(state_dict, model_name)
    print("best model is saved to ", model_name)


if __name__ == '__main__':
    cur = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    print(f'current time: {cur}')
    start = time.time()
    main()
    end = time.time()
    print(f'duration time {(end - start) / 60} mins.')
    cur = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    print(f'current time: {cur}')
