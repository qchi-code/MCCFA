import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .attention import MultiHeadAttention, MultiLayerPerceptron
from .backbone.torchvision_backbones import TVDeeplabRes101Encoder
from .prototypical_contrast import PrototypeContrastLoss


class FewShotSeg(nn.Module):
    """
    MCCFA implementation with:
      - MCA-corrected masks (provided by dataloader),
      - FDE (DCT-based frequency-domain enhancement),
      - prototype gate,
      - support-to-query aggregation,
      - foreground-background contrastive learning.

    """

    def __init__(self, use_coco_init=True):
        super().__init__()

        # Encoder now returns both spatial features and image-level threshold.
        self.encoder = TVDeeplabRes101Encoder(use_coco_init)
        self.scaler = 20.0
        self.criterion = nn.NLLLoss()

        # Feature dimension is 512 after the new backbone reduction layer.
        self.feat_dim = 512
        self.n_head = 4
        self.head_dim = self.feat_dim // self.n_head

        # Query self-attention + FFN.
        self.MHA = MultiHeadAttention(
            n_head=self.n_head,
            d_model=self.feat_dim,
            d_k=self.head_dim,
            d_v=self.head_dim,
        )
        self.MLP = MultiLayerPerceptron(dim=self.feat_dim, mlp_dim=self.feat_dim * 2)

        # Support-to-query cross-attention used by QFAT.
        self.cross_q = nn.Linear(self.feat_dim, self.feat_dim, bias=False)
        self.cross_k = nn.Linear(self.feat_dim, self.feat_dim, bias=False)
        self.cross_v = nn.Linear(self.feat_dim, self.feat_dim, bias=False)
        self.cross_proj = nn.Linear(self.feat_dim, self.feat_dim, bias=False)
        self.cross_norm = nn.LayerNorm(self.feat_dim)
        self.cross_dropout = nn.Dropout(0.1)

        # Prototype gate.
        self.gate_q = nn.Linear(self.feat_dim, self.feat_dim, bias=False)
        self.gate_s = nn.Linear(self.feat_dim, self.feat_dim, bias=False)

        # FDE hyper-parameters.
        self.fde_low_ratio = 0.15
        self.fde_high_ratio = 1.0
        self._dct_cache = {}

        # Contrastive learning.
        self.fg_sampler = np.random.RandomState(2025)
        self.fg_num = 120
        self.contrast_loss = PrototypeContrastLoss()

    def forward(self, supp_imgs, fore_mask, qry_imgs, train=False, t_loss_scaler=1):
        del t_loss_scaler  # kept only for interface compatibility

        n_ways = len(supp_imgs)
        self.n_shots = len(supp_imgs[0])
        n_queries = len(qry_imgs)
        batch_size_q = qry_imgs[0].shape[0]
        batch_size = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:]

        # ---- Encode all images once ----
        imgs_concat = torch.cat(
            [torch.cat(way, dim=0) for way in supp_imgs] + [torch.cat(qry_imgs, dim=0)],
            dim=0,
        )
        img_fts, img_thresh = self.encoder(imgs_concat, low_level=False)

        fts_size = img_fts.shape[-2:]
        supp_count = n_ways * self.n_shots * batch_size

        supp_fts = img_fts[:supp_count].view(n_ways, self.n_shots, batch_size, -1, *fts_size)
        qry_fts = img_fts[supp_count:].view(n_queries, batch_size_q, -1, *fts_size)

        supp_thresh = img_thresh[:supp_count].view(n_ways, self.n_shots, batch_size, 1)
        qry_thresh = img_thresh[supp_count:].view(n_queries, batch_size_q, 1)

        fore_mask = torch.stack([torch.stack(way, dim=0) for way in fore_mask], dim=0).float()

        outputs = []
        align_loss = img_fts.new_zeros(1)
        prototype_contrast_loss = img_fts.new_zeros(1)

        for epi in range(batch_size):
            qry_feat_epi = qry_fts[0, epi:epi + 1]
            qry_thresh_epi = qry_thresh[0, epi:epi + 1]

            fg_prototypes = []
            enhanced_qry_per_way = []
            support_maps_for_aux = []
            support_thresh_for_aux = []

            # ---- Build support-conditioned query features for each way ----
            for way in range(n_ways):
                supp_map = supp_fts[way, :, epi].mean(dim=0, keepdim=True)
                supp_mask = (fore_mask[way, :, epi].mean(dim=0, keepdim=True) > 0.5).float()
                supp_thresh_way = supp_thresh[way, :, epi].mean(dim=0, keepdim=True)

                supp_map_fde = self.apply_fde(supp_map)
                support_proto = self.getFeatures(supp_map_fde, supp_mask)

                enhanced_qry = self.ATT(qry_feat_epi, supp_map_fde, supp_mask, support_proto)

                fg_prototypes.append(support_proto)
                enhanced_qry_per_way.append(enhanced_qry)
                support_maps_for_aux.append(supp_map)
                support_thresh_for_aux.append(supp_thresh_way)

            # ---- Predict segmentation ----
            anom_s = [self.negSim(enhanced_qry_per_way[way], fg_prototypes[way]) for way in range(n_ways)]
            self.thresh_pred = [qry_thresh_epi for _ in range(n_ways)]
            pred = self.getPred(anom_s, self.thresh_pred)

            pred_ups = F.interpolate(pred, size=img_size, mode='bilinear', align_corners=True)
            pred_ups = torch.cat((1.0 - pred_ups, pred_ups), dim=1)
            outputs.append(pred_ups)

            # ---- Training losses ----
            if train:
                negative_dict = [
                    [
                        self.compute_multiple_prototypes(
                            self.fg_num,
                            supp_fts[way, shot, epi:epi + 1],
                            fore_mask[way, shot, epi:epi + 1],
                            self.fg_sampler,
                        )
                        for shot in range(self.n_shots)
                    ]
                    for way in range(n_ways)
                ]

                contrast_terms = []
                for way in range(n_ways):
                    for shot in range(self.n_shots):
                        contrast_terms.append(
                            self.contrast_loss(
                                enhanced_qry_per_way[way],
                                supp_fts[way, shot, epi:epi + 1],
                                pred,
                                fore_mask[way, shot, epi:epi + 1],
                                negative_dict[way][shot][0],
                            )
                        )
                if len(contrast_terms) > 0:
                    prototype_contrast_loss = prototype_contrast_loss + torch.stack(contrast_terms).mean()

                align_loss = align_loss + self.alignLoss(
                    enhanced_qry_per_way[0],
                    torch.cat((1.0 - pred, pred), dim=1),
                    supp_fts[:, :, epi],
                    fore_mask[:, :, epi],
                    supp_thresh[:, :, epi],
                )

        output = torch.stack(outputs, dim=1)
        output = output.view(-1, *output.shape[2:])

        if train:
            zero_thresh_loss = align_loss.detach() * 0.0
            return output, (align_loss / batch_size), zero_thresh_loss, 0.01 * (prototype_contrast_loss / batch_size)
        return output

    # ---------------------------------------------------------------------
    # QFAT components
    # ---------------------------------------------------------------------
    def ATT(self, qry_map, sup_map, mask, support_proto):
        """
        Query Feature Aggregation Transformer (QFAT).

        Args:
            qry_map: [B, C, H, W] query feature map
            sup_map: [B, C, H, W] support feature map after FDE
            mask: [B, H, W] corrected support mask
            support_proto: [B, C] support prototype for gating

        Returns:
            Enhanced query feature map of shape [B, C, H, W].
        """
        bsz, ch, h, w = qry_map.shape

        qry_tokens = qry_map.flatten(2).permute(0, 2, 1)          # [B, HW, C]
        sup_tokens = sup_map.flatten(2).permute(0, 2, 1)          # [B, HW, C]
        mask_tokens = F.interpolate(mask.unsqueeze(1), size=(h, w), mode='nearest')
        mask_tokens = mask_tokens.flatten(2).float()              # [B, 1, HW]

        # 1) query self-attention
        q_encoded = self.MHA(qry_tokens, qry_tokens, qry_tokens)  # [B, HW, C]

        # 2) prototype gate
        gate = self.compute_prototype_gate(q_encoded, support_proto)  # [B, HW, 1]

        # 3) support-to-query aggregation with column-wise normalization
        transferred = self.support_to_query_attention(q_encoded, sup_tokens, sup_tokens, mask_tokens)
        transferred = transferred * gate

        # 4) residual fusion + MLP
        fused = self.cross_norm(transferred + q_encoded)
        fused = self.MLP(fused)
        fused = fused.permute(0, 2, 1).reshape(bsz, ch, h, w)
        return fused

    def support_to_query_attention(self, q, k, v, mask_tokens=None):
        """
        Column-normalized support-to-query cross-attention.

        Each support token distributes its information over all query tokens.
        """
        bsz, len_q, _ = q.shape
        len_k = k.shape[1]

        q_proj = self.cross_q(q).view(bsz, len_q, self.n_head, self.head_dim).transpose(1, 2)
        k_proj = self.cross_k(k).view(bsz, len_k, self.n_head, self.head_dim).transpose(1, 2)
        v_proj = self.cross_v(v).view(bsz, len_k, self.n_head, self.head_dim).transpose(1, 2)

        logits = torch.matmul(q_proj, k_proj.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(logits, dim=-2)  # normalize over query tokens

        if mask_tokens is not None:
            attn = attn * mask_tokens.unsqueeze(1)

        context = torch.matmul(attn, v_proj)                       # [B, heads, HWq, d]
        context = context.transpose(1, 2).contiguous().view(bsz, len_q, self.feat_dim)
        context = self.cross_dropout(self.cross_proj(context))
        return context

    def compute_prototype_gate(self, q_tokens, support_proto):
        """
        Compute a token-wise gate from query tokens and the global support prototype.
        """
        q_gate = self.gate_q(q_tokens)                             # [B, HW, C]
        s_gate = self.gate_s(support_proto).unsqueeze(1)          # [B, 1, C]
        gate = torch.sigmoid((q_gate * s_gate).sum(dim=-1, keepdim=True))
        return gate

    # ---------------------------------------------------------------------
    # FDE (DCT-based frequency-domain enhancement)
    # ---------------------------------------------------------------------
    def apply_fde(self, feat_map):
        """
        Apply frequency-domain enhancement to the support feature map.
        """
        dct_feat = self.dct_2d(feat_map)
        band_mask = self.build_band_mask(feat_map.shape[-2:], feat_map.device, feat_map.dtype)
        filtered = dct_feat * band_mask
        enhanced = feat_map + self.idct_2d(filtered)
        return enhanced

    def build_band_mask(self, spatial_size, device, dtype):
        h, w = spatial_size
        yy = torch.linspace(0.0, 1.0, steps=h, device=device, dtype=dtype)
        xx = torch.linspace(0.0, 1.0, steps=w, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(yy, xx, indexing='ij')
        radius = torch.sqrt(yy ** 2 + xx ** 2)
        mask = ((radius >= self.fde_low_ratio) & (radius <= self.fde_high_ratio)).to(dtype)
        return mask.unsqueeze(0).unsqueeze(0)

    def _get_dct_matrix(self, n, device, dtype):
        key = (n, str(device), str(dtype))
        if key in self._dct_cache:
            return self._dct_cache[key]

        x = torch.arange(n, device=device, dtype=dtype)
        k = torch.arange(n, device=device, dtype=dtype).unsqueeze(1)
        mat = torch.cos(math.pi / n * (x + 0.5) * k)
        mat[0] = mat[0] * math.sqrt(1.0 / n)
        if n > 1:
            mat[1:] = mat[1:] * math.sqrt(2.0 / n)
        self._dct_cache[key] = mat
        return mat

    def dct_2d(self, x):
        """Channel-wise orthonormal 2D DCT-II."""
        b, c, h, w = x.shape
        dct_h = self._get_dct_matrix(h, x.device, x.dtype)
        dct_w = self._get_dct_matrix(w, x.device, x.dtype)

        x = x.reshape(b * c, h, w)
        x = torch.matmul(dct_h.unsqueeze(0), x)
        x = torch.matmul(x, dct_w.t())
        return x.reshape(b, c, h, w)

    def idct_2d(self, x):
        """Channel-wise inverse transform for the orthonormal 2D DCT."""
        b, c, h, w = x.shape
        dct_h = self._get_dct_matrix(h, x.device, x.dtype)
        dct_w = self._get_dct_matrix(w, x.device, x.dtype)

        x = x.reshape(b * c, h, w)
        x = torch.matmul(dct_h.t().unsqueeze(0), x)
        x = torch.matmul(x, dct_w)
        return x.reshape(b, c, h, w)

    # ---------------------------------------------------------------------
    # Prototype utilities and losses
    # ---------------------------------------------------------------------
    def compute_multiple_prototypes(self, fg_num, sup_fts, sup_fg, sampler):
        """
        Partition the support background into multiple sub-regions and compute
        region-level background prototypes.
        """
        bsz, ch, h, w = sup_fts.shape
        sup_fg = 1 - sup_fg
        fg_mask = F.interpolate(sup_fg.unsqueeze(0), size=sup_fts.shape[-2:], mode='bilinear', align_corners=False)
        fg_mask = fg_mask.squeeze(0).bool()
        batch_fg_protos = []

        for b in range(bsz):
            fg_protos = []
            fg_mask_i = fg_mask[b]

            with torch.no_grad():
                if fg_mask_i.sum() < fg_num:
                    fg_mask_i = fg_mask[b].clone()
                    fg_mask_i.view(-1)[:fg_num] = True

            all_centers = []
            first = True
            pts = torch.stack(torch.where(fg_mask_i), dim=1)
            for _ in range(fg_num):
                if first:
                    i = sampler.choice(pts.shape[0])
                    first = False
                else:
                    dist = pts.reshape(-1, 1, 2) - torch.stack(all_centers, dim=0).reshape(1, -1, 2)
                    i = torch.argmax((dist ** 2).sum(-1).min(1)[0])
                pt = pts[i]
                all_centers.append(pt)

            dist = pts.reshape(-1, 1, 2) - torch.stack(all_centers, dim=0).reshape(1, -1, 2)
            fg_labels = torch.argmin((dist ** 2).sum(-1), dim=1)

            fg_feats = sup_fts[b].permute(1, 2, 0)[fg_mask_i]
            for i in range(fg_num):
                proto = fg_feats[fg_labels == i].mean(0)
                fg_protos.append(proto)

            fg_protos = torch.stack(fg_protos, dim=1)
            batch_fg_protos.append(fg_protos)

        fg_proto = torch.stack(batch_fg_protos, dim=0).transpose(1, 2)
        return fg_proto

    def getFeatures_FU(self, fts, mask):
        """Collect background pixel features. Kept for compatibility."""
        fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear', align_corners=True)
        b_list = []
        for i in range(mask.size(1)):
            for j in range(mask.size(2)):
                if mask[0, i, j] == 0:
                    b_part = fts[0, :, i, j].unsqueeze(0)
                    b_list.append(b_part)
        combined_tensor = torch.cat(b_list, dim=0)
        return combined_tensor

    def getFeatures_all(self, fts):
        """Flatten all spatial features into token form."""
        fts_ = fts.squeeze(0).permute(1, 2, 0)
        fts_ = fts_.view(fts_.size(0) * fts_.size(1), fts_.size(2))
        return fts_

    def negSim(self, fts, prototype):
        """
        Calculate negative cosine similarity between features and a prototype.
        """
        sim = -F.cosine_similarity(fts, prototype[..., None, None], dim=1) * self.scaler
        return sim

    def getFeatures(self, fts, mask):
        """
        Extract a masked prototype via masked average pooling.
        """
        fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear', align_corners=False)
        masked_fts = torch.sum(fts * mask.unsqueeze(1), dim=(2, 3)) / (mask.sum(dim=(1, 2), keepdim=False).unsqueeze(1) + 1e-5)
        return masked_fts

    def getPrototype(self, fg_fts):
        """Average shot-level features to obtain class prototypes."""
        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        fg_prototypes = [
            torch.sum(torch.cat([tr for tr in way], dim=0), dim=0, keepdim=True) / n_shots
            for way in fg_fts
        ]
        return fg_prototypes

    def alignLoss(self, qry_fts, pred, supp_fts, fore_mask, supp_thresh):
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])

        pred_mask = pred.argmax(dim=1, keepdim=True)
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]
        skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        pred_mask = torch.stack(binary_masks, dim=1).float()
        qry_prototypes = torch.sum(qry_fts.unsqueeze(1) * pred_mask, dim=(0, 3, 4))
        qry_prototypes = qry_prototypes / (pred_mask.sum((0, 3, 4)) + 1e-5)

        loss = qry_fts.new_zeros(1)
        for way in range(n_ways):
            if way in skip_ways:
                continue
            for shot in range(n_shots):
                img_fts = supp_fts[way, [shot]]
                supp_sim = self.negSim(img_fts, qry_prototypes[[way + 1]])

                pred = self.getPred([supp_sim], [supp_thresh[way, shot:shot + 1]])
                pred_ups = F.interpolate(pred, size=fore_mask.shape[-2:], mode='bilinear', align_corners=True)
                pred_ups = torch.cat((1.0 - pred_ups, pred_ups), dim=1)

                supp_label = torch.full_like(fore_mask[way, shot], 255, device=img_fts.device)
                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[fore_mask[way, shot] == 0] = 0

                eps = torch.finfo(torch.float32).eps
                log_prob = torch.log(torch.clamp(pred_ups, eps, 1 - eps))
                loss += self.criterion(log_prob, supp_label[None, ...].long()) / n_shots / n_ways

        return loss

    def getPred(self, sim, thresh):
        """
        Convert similarity maps into foreground probabilities.
        """
        pred = []
        for s, t in zip(sim, thresh):
            t = t.view(-1, 1, 1)
            pred.append(1.0 - torch.sigmoid(s - t))
        return torch.stack(pred, dim=1)
