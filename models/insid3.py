"""INSID3: In-context segmentation with a frozen DINOv3 encoder."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from utils.clustering import agglomerative_clustering, compute_cluster_prototypes
from utils.data import build_transform, downsample_mask
from utils.refinement import upsample_mask, init_crf, crf_refine
import math
import numpy as np

from PIL import Image
from typing import Union, Tuple, List

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class INSID3(nn.Module):
    """Training-free in-context segmentation using a frozen DINOv3 encoder."""

    def __init__(
        self,
        encoder: nn.Module,
        image_size: Union[int, list, tuple] = 1024,
        svd_components: int = 500,
        tau: float = 0.6,
        merge_threshold: float = 0.2,
        mask_refiner: str = "bilinear",
        resize_to_orig_size: bool = True,
        device: str = DEVICE,
    ):
        super().__init__()
        self.encoder = encoder
        self.image_size = (image_size, image_size) if isinstance(image_size, int) else image_size
        self.svd_components = svd_components
        self.tau = tau
        self.merge_threshold = merge_threshold
        self.mask_refiner = mask_refiner
        self.resize_to_orig_size = resize_to_orig_size
        self.device = device
        self.positional_basis = self._build_positional_basis(device)

        if mask_refiner == 'crf':
            self._crf, self._crf_band_px, self._crf_p_core = init_crf(image_size, device)

        self._transform = build_transform(image_size)
        self._ref_images = None
        self._ref_masks = None
        self._tgt_image = None
        self._orig_tgt_size = None

        # sim maps
        self._sim_maps = None
        self._deb_sim_maps = None

        # patch size of DINOv3 = 16
        self._patch_size = 16

    def reset(self):
        self._ref_images = None
        self._ref_masks = None
        self._tgt_image = None
        self._orig_tgt_size = None
        self._sim_maps = None
        self._deb_sim_maps = None
        return

    def set_reference(self, image: str | 'Image.Image', mask: str | 'Image.Image' | torch.Tensor) -> None:
        """Set reference image and mask from file paths or PIL Images.

        Args:
            image: path (str) or PIL Image.
            mask:  path (str), PIL Image, or torch.Tensor (binary / grayscale).
        """
        # Handle image
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        # Handle mask
        if isinstance(mask, torch.Tensor):
            mask_tensor = mask.unsqueeze(0) if mask.ndim == 2 else mask  # (H,W) or (1,H,W)
            if mask_tensor.dtype != torch.bool:
                mask_tensor = mask_tensor > 0
        else:
            if isinstance(mask, str):
                mask = Image.open(mask)
            mask_tensor = torch.tensor(
                np.array(mask) > 0, dtype=torch.bool
            ).unsqueeze(0)

        if self._ref_images is None:
            self._ref_images = [image]
            self._ref_masks = [mask_tensor]
        else:
            self._ref_images = self._ref_images.append(image)
            self._ref_masks = self._ref_masks.append(mask)

    def set_target(self, image: str | 'Image.Image') -> None:
        """Set target image from a file path or PIL Image.

        Args:
            image: path (str) or PIL Image.
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        self._orig_tgt_size = (image.height, image.width)
        self._tgt_image = image

    def segment(self) -> torch.Tensor:
        """Run prediction using previously set reference(s) and target.

        Returns:
            pred_mask: (H, W) boolean mask at original target resolution.
        """
        def handle_mask(mask_tensor: torch.Tensor):
            # Resize mask to model resolution
            mask_tensor = F.interpolate(
                mask_tensor.unsqueeze(0).float(),
                size=self.image_size, mode='nearest',
            ).squeeze(0) > 0.5
            return mask_tensor
        
        # ref image & ref mask
        ref_img_tensor = self._transform(self._ref_images[0])
        ref_img_tensor = ref_img_tensor.unsqueeze(0) # -> [1,C,H,W]
        ref_mask_tensor = handle_mask(self._ref_masks[0]) # -> [1,H,W]

        for n in range(1, len(self._ref_images)):
            img_tensor = self._transform(self._ref_images[n])
            img_tensor = img_tensor.unsqueeze(0)
            mask_tensor = handle_mask(self._ref_masks[n])
            # concat
            ref_img_tensor = torch.cat([ref_img_tensor, img_tensor], dim=0)
            ref_mask_tensor = torch.cat([ref_mask_tensor, mask_tensor], dim=0)
        # move to device
        ref_img_tensor = ref_img_tensor.to(self.device) # [S,C,H,W]
        ref_mask_tensor = ref_mask_tensor.to(self.device) # [S,H,W]
        # target image
        tgt_img_tensor = self._transform(self._tgt_image).to(self.device)
        pred = self.predict(ref_img_tensor, ref_mask_tensor, tgt_img_tensor)
        return pred

    @torch.no_grad()
    def predict(self, ref_images: torch.Tensor, ref_masks: torch.Tensor, tgt_image: torch.Tensor) -> torch.Tensor:
        """Segment the target image given reference image(s) and mask(s).

        Args:
            ref_images: (S, C, H, W) reference image(s).
            ref_masks:  (S, H, W) binary reference mask(s).
            tgt_image:  (C, H, W) target image.

        Returns:
            pred_mask: (H, W) boolean mask at input resolution.
        """
        S = ref_images.shape[0]
        tgt_image = tgt_image.unsqueeze(0)  # (C, H, W) → (1, C, H, W)
        imgs = torch.cat([ref_images, tgt_image], dim=0).unsqueeze(0)

        # Feature extraction
        fmaps = self._extract_features(imgs)
        fmaps_norm = F.normalize(fmaps, p=2, dim=2)
        _, _, C, h, w = fmaps_norm.shape

        ref_masks = ref_masks.unsqueeze(1)
        feat_tgt = fmaps_norm[:, S]

        # Positional debiasing
        fmaps_debiased = self._debias_features(fmaps_norm)
        feat_refs_deb = fmaps_debiased[:, :S]
        feat_tgt_deb = fmaps_debiased[:, S]

        # Reference prototype (averaged across shots)
        ref_prototypes = []
        for s in range(S):
            mask_s = downsample_mask(ref_masks[s:s+1], h, w)
            fg = feat_refs_deb[0, s, :, mask_s]
            if fg.shape[1] > 0:
                ref_prototypes.append(fg.mean(dim=1))
        ref_prototype = F.normalize(
            torch.stack(ref_prototypes).mean(dim=0), p=2, dim=0
        ).unsqueeze(1)

        # Candidate localization (forward + backward matching)
        # Compute similarity maps between each reference and the target (debiased space)
        sim_maps = []
        for m in range(S):
            feat_ref_m = feat_refs_deb[:, m]
            sim_m = torch.einsum('bchw,bcxy->bhwxy', feat_ref_m, feat_tgt_deb)
            sim_maps.append(sim_m)
        candidate_mask = self._locate_candidates(
            sim_maps, ref_masks, feat_tgt_deb, ref_prototype, h, w
        )
        if candidate_mask.sum() == 0:
            return self._finalize_mask(candidate_mask, tgt_image)

        # Fine-grained clustering
        feat_tgt_flat = feat_tgt[0].reshape(C, -1).permute(1, 0)
        cluster_labels = agglomerative_clustering(feat_tgt_flat, self.tau).reshape(h, w)
        K = int(cluster_labels.max().item()) + 1

        feat_tgt_deb_flat = feat_tgt_deb[0].reshape(C, -1).permute(1, 0)
        cluster_protos = compute_cluster_prototypes(
            feat_tgt_deb_flat, cluster_labels.view(-1), K
        )

        # Seed selection and cluster aggregation
        pred_mask = self._seed_and_aggregate(
            candidate_mask, cluster_labels, cluster_protos, K,
            ref_prototype, feat_tgt, feat_tgt_deb, h, w
        )
        return self._finalize_mask(pred_mask, tgt_image)

    @torch.no_grad()
    def get_sim_maps(self) -> Tuple[List[torch.Tensor]]:
        """Calculate similarity maps.

        Returns:
            sim_maps: (sim_maps, debiased_sim_maps)
        """
        if self._sim_maps is not None and self._deb_sim_maps is not None:
            return self._sim_maps, self._deb_sim_maps

        def get_patch_divisible_size(img):
            if isinstance(img, Image.Image):
                H, W = img.height, img.width
            elif isinstance(img, np.ndarray):
                H, W, C = img.shape
            elif isinstance(img, torch.Tensor):
                C, H, W = img.shape
            else:
                raise TypeError
            patch_h = int(H // self._patch_size) # num of patch along height
            patch_w = int(W // self._patch_size) # num of patch along width
            new_H = patch_h * self._patch_size
            new_W = patch_w * self._patch_size
            return (new_H, new_W)
        
        h, w = get_patch_divisible_size(self._tgt_image)
        tgt_img_transform = build_transform(image_size=(h, w))
        tgt_img = tgt_img_transform(self._tgt_image)
        tgt_img = tgt_img.unsqueeze(0).unsqueeze(0) # (C,H,W) → (1,1,C,H,W)
        tgt_img = tgt_img.to(self.device)

        bias_sim_maps = []
        deb_sim_maps = []

        S = len(self._ref_images)
        for s in range(S):
            ref_img = self._ref_images[s]
            h, w = get_patch_divisible_size(ref_img)
            ref_img_transform = build_transform(image_size=(h,w))
            ref_img = ref_img_transform(ref_img)
            ref_img = ref_img.unsqueeze(0).unsqueeze(0) # ->(1,1,C,H,W)           

            # Feature extraction
            ref_img = ref_img.to(self.device)

            ref_fmaps = self._extract_features(ref_img)
            tgt_fmaps = self._extract_features(tgt_img)

            ref_fmaps_norm = F.normalize(ref_fmaps, p=2, dim=2)
            tgt_fmaps_norm = F.normalize(tgt_fmaps, p=2, dim=2)

            feat_ref = ref_fmaps_norm[:, :] # -> [1,C,H,W]
            feat_tgt = tgt_fmaps_norm[:, 0] # -> [1,C,H,W]

            # Positional debiasing
            feat_refs_deb = self._debias_features(ref_fmaps_norm)
            feat_tgt_deb = self._debias_features(tgt_fmaps_norm)
            feat_refs_deb = feat_refs_deb[:, 0] # -> [1,C,H,W]
            feat_tgt_deb = feat_tgt_deb[:, 0]  # -> [1,C,H,W]

            # Candidate localization (forward + backward matching)
            # Compute similarity maps between each reference and the target (debiased space)
            # biased & debiased
            bias_sim_m = torch.einsum('bchw,bcxy->bhwxy', feat_refs_deb, feat_tgt)
            deb_sim_m = torch.einsum('bchw,bcxy->bhwxy', feat_refs_deb, feat_tgt_deb)
            bias_sim_maps.append(bias_sim_m)
            deb_sim_maps.append(deb_sim_m)
        self._sim_maps = bias_sim_maps
        self._deb_sim_maps = deb_sim_maps
        return bias_sim_maps, deb_sim_maps

    # ──────── Feature extraction ────────

    def _extract_features(self, imgs: torch.Tensor) -> torch.Tensor:
        B, T = imgs.shape[:2]
        x = einops.rearrange(imgs, 'b t c h w -> (b t) c h w')
        fmaps = self.encoder.get_intermediate_layers(x, n=1, reshape=True)[0]
        return einops.rearrange(fmaps, '(b t) c h w -> b t c h w', b=B)

    # ──────── Positional debiasing ────────

    @torch.no_grad()
    def _build_positional_basis(self, device: str) -> torch.Tensor:
        """Estimate the positional subspace from a noise image via SVD."""
        from torchvision.transforms.functional import normalize
        H, W = self.image_size[0], self.image_size[1]
        noise_img = normalize(
            torch.zeros(1, 3, H, W),
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
        ).to(device)
        noise_fmaps = self.encoder.to(device).get_intermediate_layers(
            noise_img, n=1, reshape=True
        )[0]
        noise_fmaps = F.normalize(noise_fmaps, p=2, dim=1)

        E = einops.rearrange(noise_fmaps, 'b c h w -> c (b h w)')
        E = E - E.mean(dim=1, keepdim=True)
        U, _, _ = torch.linalg.svd(E, full_matrices=False)
        return U[:, :self.svd_components].contiguous()

    def _debias_features(self, fmaps_norm: torch.Tensor) -> torch.Tensor:
        """Project features onto the orthogonal complement of the positional subspace."""
        B, T, C, H, W = fmaps_norm.shape
        X = fmaps_norm.reshape(B * T, C, H * W)

        basis = self.positional_basis.to(X.device)
        P_perp = torch.eye(C, device=X.device, dtype=X.dtype) - basis @ basis.T
        X_deb = torch.matmul(P_perp.unsqueeze(0), X).reshape(B, T, C, H, W)
        return F.normalize(X_deb, p=2, dim=2)

    # ──────── Candidate localization ────────

    def _locate_candidates(
        self,
        sim_maps: list,
        ref_masks: torch.Tensor,
        feat_tgt_deb: torch.Tensor,
        ref_prototype: torch.Tensor,
        h: int,
        w: int,
    ) -> torch.Tensor:
        """Find candidate target patches via forward and backward matching."""
        # Forward: positive similarity to aggregated reference prototype
        sim_fwd = torch.einsum('bchw,cd->bhw', feat_tgt_deb, ref_prototype).squeeze(0)
        forward_mask = sim_fwd > 0
        if forward_mask.sum() == 0:
            forward_mask = sim_fwd > float(torch.quantile(sim_fwd, 0.9))

        # Backward: majority-vote over nearest neighbours in each reference
        k = len(sim_maps)
        votes = torch.zeros((h, w), dtype=torch.int32, device=sim_maps[0].device)
        for m, sim_m in enumerate(sim_maps):
            sim0 = sim_m[0]  # (Hs, Ws, h, w)
            Hs, Ws = sim0.shape[:2]
            sim_t_to_r = sim0.permute(2, 3, 0, 1)  # (h, w, Hs, Ws)
            best_idx = sim_t_to_r.reshape(h, w, -1).argmax(dim=2)  # (h, w)
            rows = best_idx // Ws
            cols = best_idx % Ws
            ref_mask_m = downsample_mask(ref_masks[m:m+1], Hs, Ws).squeeze(0)  # (Hs, Ws)
            votes += ref_mask_m[rows, cols].to(torch.int32)

        majority_thresh = math.ceil(k / 2)
        backward_mask = votes >= majority_thresh

        return forward_mask & backward_mask

    # ──────── Seed selection and cluster aggregation ────────

    def _seed_and_aggregate(
        self,
        candidate_mask: torch.Tensor,
        cluster_labels: torch.Tensor,
        cluster_protos: torch.Tensor,
        K: int,
        ref_prototype: torch.Tensor,
        feat_tgt: torch.Tensor,
        feat_tgt_deb: torch.Tensor,
        h: int,
        w: int,
    ) -> torch.Tensor:
        """Select the seed cluster and aggregate remaining clusters."""
        matched_mask = candidate_mask & (cluster_labels >= 0)
        if matched_mask.sum() == 0:
            return candidate_mask

        matched_ids, n_pixels = cluster_labels[matched_mask].unique(return_counts=True)

        # Area weighting
        all_areas = cluster_labels[cluster_labels >= 0].unique(return_counts=True)[1]
        per_cluster = torch.zeros(K, device=cluster_labels.device)
        per_cluster[matched_ids] = n_pixels.float()
        area_weights = per_cluster / all_areas

        # Seed selection: cluster with highest cross-image similarity
        protos_matched = cluster_protos[matched_ids]
        cross_sim_matched = (ref_prototype.T @ protos_matched.T).squeeze(0)
        seed_idx = int(torch.argmax(cross_sim_matched).item())
        seed_id = matched_ids[seed_idx].item()

        # Intra-image similarity to seed (original feature space)
        feat_tgt_flat = feat_tgt[0].reshape(feat_tgt.shape[1], -1).permute(1, 0)
        orig_protos = compute_cluster_prototypes(
            feat_tgt_flat, cluster_labels.view(-1), K
        )
        intra_sim = torch.einsum('c,kc->k', orig_protos[seed_id], orig_protos)

        # Cross-image similarity per cluster (debiased space)
        fg_sim = torch.einsum('bchw,cd->bhw', feat_tgt_deb, ref_prototype).squeeze(0)
        cross_sim = torch.empty(K, device=fg_sim.device, dtype=fg_sim.dtype)
        for k in range(K):
            idx = (cluster_labels == k)
            cross_sim[k] = fg_sim[idx].mean() if idx.any() else 0.0

        # Combined score
        combined = cross_sim * intra_sim
        area_weights[seed_id] = 1.0
        combined *= area_weights

        final_mask = torch.zeros(h, w, dtype=torch.bool, device=cluster_labels.device)
        valid = cluster_labels >= 0
        final_mask[valid] = combined[cluster_labels[valid]] > self.merge_threshold
        return final_mask

    # ──────── Mask finalization ────────

    def _finalize_mask(self, mask: torch.Tensor, tgt_image: torch.Tensor) -> torch.Tensor:
        """Upsample feature-resolution mask, optionally with CRF refinement."""
        H, W = tgt_image.shape[-2:]
        up = upsample_mask(mask, H, W)
        if self.mask_refiner == 'crf':
            up = crf_refine(self._crf, self._crf_band_px, self._crf_p_core, tgt_image, up)
        # Resize to original target resolution
        if self.resize_to_orig_size:
            up = upsample_mask(up, self._orig_tgt_size[0], self._orig_tgt_size[1])
        return up
