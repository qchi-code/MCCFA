import PIL
import cv2
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
import torchvision.transforms as deftfx
import glob
import os
import SimpleITK as sitk
import random
import numpy as np
from collections import deque
import utils
from . import image_transforms as myit
from .dataset_specifics import *


class TestDataset(Dataset):
    """
    Dataset for episodic evaluation in few-shot medical image segmentation.

    This dataset loads the test fold only, selects one support volume according to
    args.supp_idx, and uses the remaining volumes as query cases.
    """

    def __init__(self, args):
        """
        Initialize the test dataset.

        Args:
            args: Argument container with dataset name, root path, fold index,
                  support index, and evaluation protocol flag.
        """
        # Read image paths for the selected dataset.
        if args.dataset == 'CMR':
            self.image_dirs = glob.glob(os.path.join(args.data_root, 'cmr_MR_normalized/image*'))
        elif args.dataset == 'CHAOST2':
            self.image_dirs = glob.glob(os.path.join(args.data_root, 'chaos_MR_T2_normalized/image*'))
        elif args.dataset == 'SABS':
            self.image_dirs = glob.glob(os.path.join(args.data_root, 'sabs_CT_normalized/image*'))

        self.image_dirs = sorted(
            self.image_dirs,
            key=lambda x: int(x.split('_')[-1].split('.nii.gz')[0])
        )

        # Keep only the test fold.
        self.FOLD = get_folds(args.dataset)
        self.image_dirs = [elem for idx, elem in enumerate(self.image_dirs) if idx in self.FOLD[args.fold]]

        # Split the test fold into one support volume and the remaining query volumes.
        idx = np.arange(len(self.image_dirs))
        self.support_dir = self.image_dirs[idx[args.supp_idx]]
        self.image_dirs.pop(idx[args.supp_idx])
        self.label = None

        # Evaluation protocol flag.
        self.EP1 = args.EP1

    def __len__(self):
        """Return the number of query volumes."""
        return len(self.image_dirs)

    def __getitem__(self, idx):
        """
        Load one query volume and its binary label map for the current target class.

        Args:
            idx: Query volume index.

        Returns:
            dict: A sample containing:
                - 'id': image path
                - 'image': tensor of shape [D, 3, H, W]
                - 'label': tensor of shape [D, H, W]
        """
        img_path = self.image_dirs[idx]

        img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
        img = (img - img.mean()) / img.std()
        img = np.stack(3 * [img], axis=1)

        lbl = sitk.GetArrayFromImage(
            sitk.ReadImage(img_path.split('image_')[0] + 'label_' + img_path.split('image_')[-1])
        )
        lbl[lbl == 200] = 1
        lbl[lbl == 500] = 2
        lbl[lbl == 600] = 3
        lbl = 1 * (lbl == self.label)

        sample = {'id': img_path}

        # Evaluation Protocol 1:
        # keep only slices that contain the target organ.
        if self.EP1:
            idx = lbl.sum(axis=(1, 2)) > 0
            sample['image'] = torch.from_numpy(img[idx])
            sample['label'] = torch.from_numpy(lbl[idx])

        # Evaluation Protocol 2:
        # keep the full volume regardless of target presence in each slice.
        else:
            sample['image'] = torch.from_numpy(img)
            sample['label'] = torch.from_numpy(lbl)

        return sample

    def get_support_index(self, n_shot, C):
        """
        Select support slice indices according to the protocol of Ouyang et al.

        Args:
            n_shot: Number of support slices.
            C: Number of available positive slices.

        Returns:
            np.ndarray: Selected slice indices.
        """
        if n_shot == 1:
            pcts = [0.5]
        else:
            half_part = 1 / (n_shot * 2)
            part_interval = (1.0 - 1.0 / n_shot) / (n_shot - 1)
            pcts = [half_part + part_interval * ii for ii in range(n_shot)]

        return (np.array(pcts) * C).astype('int')

    def getSupport(self, label=None, all_slices=True, N=None):
        """
        Load the support volume and return either all slices or selected labeled slices.

        Args:
            label: Target class label.
            all_slices: If True, return the full support volume.
            N: Number of labeled support slices to sample when all_slices is False.

        Returns:
            dict: A support sample containing:
                - 'image': support image tensor
                - 'label': support binary mask tensor
        """
        if label is None:
            raise ValueError('Need to specify label class!')

        img_path = self.support_dir
        img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
        img = (img - img.mean()) / img.std()
        img = np.stack(3 * [img], axis=1)

        lbl = sitk.GetArrayFromImage(
            sitk.ReadImage(img_path.split('image_')[0] + 'label_' + img_path.split('image_')[-1])
        )

        # Remap labels when needed.
        lbl[lbl == 200] = 1
        lbl[lbl == 500] = 2
        lbl[lbl == 600] = 3
        lbl = 1 * (lbl == label)

        sample = {}
        if all_slices:
            sample['image'] = torch.from_numpy(img)
            sample['label'] = torch.from_numpy(lbl)
        else:
            # Select N labeled slices from the support volume.
            if N is None:
                raise ValueError('Need to specify number of labeled slices!')
            idx = lbl.sum(axis=(1, 2)) > 0
            idx_ = self.get_support_index(N, idx.sum())

            sample['image'] = torch.from_numpy(img[idx][idx_])
            sample['label'] = torch.from_numpy(lbl[idx][idx_])

        return sample


class TrainDataset(Dataset):
    """
    Dataset for self-supervised episodic training.

    Each episode samples one supervoxel class from one training volume, then
    constructs support and query slices from consecutive slices containing that class.
    """

    def __init__(self, args):
        """
        Initialize the training dataset.

        Args:
            args: Argument container with dataset settings, fold index, number of
                  shots, number of queries, supervoxel count, and max iterations.
        """
        self.n_shot = args.n_shot
        self.n_way = args.n_way
        self.n_query = args.n_query
        self.n_sv = args.n_sv
        self.max_iter = args.max_iterations
        self.read = True
        self.train_sampling = 'neighbors'
        self.min_size = 200

        # Read image paths and supervoxel paths.
        if args.dataset == 'CMR':
            self.image_dirs = glob.glob(os.path.join(args.data_root, 'cmr_MR_normalized/image*'))
        elif args.dataset == 'CHAOST2':
            self.image_dirs = glob.glob(os.path.join(args.data_root, 'chaos_MR_T2_normalized/image*'))
        elif args.dataset == 'SABS':
            self.image_dirs = glob.glob(os.path.join(args.data_root, 'sabs_CT_normalized/image*'))

        self.image_dirs = sorted(
            self.image_dirs,
            key=lambda x: int(x.split('_')[-1].split('.nii.gz')[0])
        )

        self.sprvxl_dirs = glob.glob(os.path.join(args.data_root, 'supervoxels_' + str(args.n_sv), 'super*'))
        self.sprvxl_dirs = sorted(
            self.sprvxl_dirs,
            key=lambda x: int(x.split('_')[-1].split('.nii.gz')[0])
        )

        # Remove the test fold and keep only training data.
        self.FOLD = get_folds(args.dataset)
        self.image_dirs = [elem for idx, elem in enumerate(self.image_dirs) if idx not in self.FOLD[args.fold]]
        self.sprvxl_dirs = [elem for idx, elem in enumerate(self.sprvxl_dirs) if idx not in self.FOLD[args.fold]]

        # Optionally preload volumes into memory for faster episodic sampling.
        if self.read:
            self.images = {}
            self.sprvxls = {}
            for image_dir, sprvxl_dir in zip(self.image_dirs, self.sprvxl_dirs):
                self.images[image_dir] = sitk.GetArrayFromImage(sitk.ReadImage(image_dir))
                self.sprvxls[sprvxl_dir] = sitk.GetArrayFromImage(sitk.ReadImage(sprvxl_dir))

    def __len__(self):
        """
        Return the number of training iterations.

        Note:
            This dataset is episode-based. The returned length corresponds to the
            number of episodes rather than the number of unique volumes.
        """
        return self.max_iter

    def gamma_tansform(self, img):
        """
        Apply random gamma intensity augmentation.

        Args:
            img: Input image array.

        Returns:
            np.ndarray: Gamma-transformed image.
        """
        gamma_range = (0.5, 1.5)
        gamma = np.random.rand() * (gamma_range[1] - gamma_range[0]) + gamma_range[0]
        cmin = img.min()
        irange = (img.max() - cmin + 1e-5)

        img = img - cmin + 1e-5
        img = irange * np.power(img * 1.0 / irange, gamma)
        img = img + cmin

        return img

    def geom_transform(self, img, mask):
        """
        Apply geometric augmentation jointly to images and masks.

        The transformation includes affine deformation and elastic deformation.

        Args:
            img: Input image array.
            mask: Corresponding binary mask array.

        Returns:
            tuple: Transformed (img, mask).
        """
        affine = {'rotate': 5, 'shift': (5, 5), 'shear': 5, 'scale': (0.9, 1.2)}
        alpha = 10
        sigma = 5
        order = 3

        tfx = []
        tfx.append(
            myit.RandomAffine(
                affine.get('rotate'),
                affine.get('shift'),
                affine.get('shear'),
                affine.get('scale'),
                affine.get('scale_iso', True),
                order=order
            )
        )
        tfx.append(myit.ElasticTransform(alpha, sigma))
        transform = deftfx.Compose(tfx)

        if len(img.shape) > 4:
            n_shot = img.shape[1]
            for shot in range(n_shot):
                cat = np.concatenate((img[0, shot], mask[:, shot])).transpose(1, 2, 0)
                cat = transform(cat).transpose(2, 0, 1)
                img[0, shot] = cat[:3, :, :]
                mask[:, shot] = np.rint(cat[3:, :, :])

        else:
            for q in range(img.shape[0]):
                cat = np.concatenate((img[q], mask[q][None])).transpose(1, 2, 0)
                cat = transform(cat).transpose(2, 0, 1)
                img[q] = cat[:3, :, :]
                mask[q] = np.rint(cat[3:, :, :].squeeze())

        return img, mask

    def __getitem__(self, idx):
        """
        Sample one training episode.

        Workflow:
            1. Randomly choose one training volume.
            2. Randomly choose one supervoxel class inside that volume.
            3. Collect consecutive slices containing that class.
            4. Apply largest-connected-component filtering to each sampled slice.
            5. Build support/query image-mask pairs.
            6. Apply random intensity/geometric augmentation.

        Args:
            idx: Dummy index required by PyTorch Dataset.

        Returns:
            dict: Episode sample containing support/query images and labels.
        """
        # Randomly sample one patient/volume.
        pat_idx = random.choice(range(len(self.image_dirs)))

        if self.read:
            img = self.images[self.image_dirs[pat_idx]]
            sprvxl = self.sprvxls[self.sprvxl_dirs[pat_idx]]
        else:
            img = sitk.GetArrayFromImage(sitk.ReadImage(self.image_dirs[pat_idx]))
            sprvxl = sitk.GetArrayFromImage(sitk.ReadImage(self.sprvxl_dirs[pat_idx]))

        # Normalize the image volume.
        img = (img - img.mean()) / img.std()

        # Sample one supervoxel label excluding background (label 0).
        unique = list(np.unique(sprvxl))
        unique.remove(0)

        size = 0
        while size < self.min_size:
            n_slices = (self.n_shot * self.n_way) + self.n_query - 1

            # Ensure the sampled class appears in enough slices.
            while n_slices < ((self.n_shot * self.n_way) + self.n_query):
                cls_idx = random.choice(unique)
                sli_idx = np.sum(sprvxl == cls_idx, axis=(1, 2)) > 0
                n_slices = np.sum(sli_idx)

            img_slices = img[sli_idx]
            sprvxl_slices = 1 * (sprvxl[sli_idx] == cls_idx)

            # Sample consecutive slices for support/query construction.
            i = random.choice(
                np.arange(n_slices - ((self.n_shot * self.n_way) + self.n_query) + 1)
            )
            sample = np.arange(i, i + (self.n_shot * self.n_way) + self.n_query)

            # Apply largest connected component filtering slice by slice.
            for i in sample:
                sprvxl_slices[i] = self.largest_connected_component(sprvxl_slices[i])
                size = np.sum(sprvxl_slices[i])
                if size < self.min_size:
                    break

        # Randomly reverse the order of consecutive slices.
        if np.random.random(1) > 0.5:
            sample = sample[::-1]

        # Build support/query masks.
        sup_lbl = sprvxl_slices[sample[:self.n_shot * self.n_way]][None,]
        qry_lbl = sprvxl_slices[sample[self.n_shot * self.n_way:]]

        # Build support/query images and expand them to 3 channels.
        sup_img = img_slices[sample[:self.n_shot * self.n_way]][None,]
        sup_img = np.stack((sup_img, sup_img, sup_img), axis=2)
        qry_img = img_slices[sample[self.n_shot * self.n_way:]]
        qry_img = np.stack((qry_img, qry_img, qry_img), axis=1)

        # Apply gamma augmentation to either support or query.
        if np.random.random(1) > 0.5:
            qry_img = self.gamma_tansform(qry_img)
        else:
            sup_img = self.gamma_tansform(sup_img)

        # Apply geometric augmentation to either support or query.
        if np.random.random(1) > 0.5:
            qry_img, qry_lbl = self.geom_transform(qry_img, qry_lbl)
        else:
            sup_img, sup_lbl = self.geom_transform(sup_img, sup_lbl)

        sample = {
            'support_images': sup_img,
            'support_fg_labels': sup_lbl,
            'query_images': qry_img,
            'query_labels': qry_lbl
        }

        return sample

    def largest_connected_component(self, a):
        """
        Keep only the largest 4-connected component in a binary mask.

        This function is used as the mask correction step during training to
        suppress scattered discrete fragments in pseudo-masks.

        Args:
            a: Binary 2D numpy array.

        Returns:
            np.ndarray: Binary mask containing only the largest connected region.
        """
        rows, cols = a.shape
        visited = np.zeros_like(a, dtype=np.bool_)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        def bfs(start_r, start_c):
            """
            Breadth-first search for one connected component.

            Args:
                start_r: Start row index.
                start_c: Start column index.

            Returns:
                tuple: (component_size, component_pixels)
            """
            queue = deque([(start_r, start_c)])
            current_size = 0
            component = []

            while queue:
                r, c = queue.popleft()
                if visited[r, c]:
                    continue
                visited[r, c] = True
                current_size += 1
                component.append((r, c))

                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and a[nr, nc] == 1 and not visited[nr, nc]:
                        queue.append((nr, nc))

            return current_size, component

        max_size = 0
        max_component = []

        for r in range(rows):
            for c in range(cols):
                if a[r, c] == 1 and not visited[r, c]:
                    size, component = bfs(r, c)
                    if size > max_size:
                        max_size = size
                        max_component = component

        result = np.zeros_like(a)
        for r, c in max_component:
            result[r, c] = 1

        return result