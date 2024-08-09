# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 11:51:04 2022

@author: Yucheng
"""

import torch
import torchio as tio
import os
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms
import torchio.transforms as T
import matplotlib.pyplot as plt


class AortaDataset:
    def __init__(self, data_root, transform=None, size=(512, 512), return_meta=False):
        label_path = os.path.join(data_root, "labels")
        image_path = os.path.join(data_root, "imgs")

        self.transform = transform
        self.return_meta = return_meta
        self.label_files = sorted(os.listdir(label_path))
        self.label_list = [os.path.join(label_path, file) for file in self.label_files]
        self.image_files = sorted(os.listdir(image_path))
        # self.image_list = [os.path.join(image_path, file) for file in self.image_files]
        self.image_list = [os.path.join(image_path, file) for file in self.image_files if
                           os.path.splitext(file)[0] in [os.path.splitext(os.path.splitext(i)[0])[0] for i in
                                                         self.label_files]]
        self.resizor = torchvision.transforms.Resize(size)
        self.transform = transform

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label_map = self.label_list[idx]
        lab = tio.LabelMap(label_map).tensor
        # find slice with annotation (2D)
        slice_idx = lab.max(dim=3)[1].max()
        # select label slice (2D)
        label = torch.select(lab, dim=3, index=slice_idx)
        label = self.resizor(label)
        # label = label.type(torch.t)

        img_idx = self.get_image_dix(idx)
        nii_image = self.image_list[img_idx]
        dat = tio.ScalarImage(nii_image).tensor.to(torch.float)
        # find corresponding slice in image
        data = torch.select(dat, dim=3, index=slice_idx)
        data = self.resizor(data)

        if self.transform:
            transformed = self.transform({'scan': data.unsqueeze(-1),
                                          'annot': label.unsqueeze(-1)})
            data = transformed['scan'].squeeze(3).to(data.dtype)
            label = transformed['annot'].squeeze(3).to(label.dtype)
        if self.return_meta:
            scan_name = os.path.basename(self.label_list[idx])
            scan_name = scan_name.split(".")[0]
            scan_name = scan_name.split("%")[0]
            return data, label, torch.empty((0, 0)), {"scan_name": scan_name}
        else:
            return data, label, torch.empty((0, 0))

    def get_image_dix(self, idx):
        a = [i.split(".")[0] for i in self.image_files]
        b = [i.split("%")[0][:7] for i in self.label_files]

        return a.index(b[idx])


class AortaDatasetInterpolation(Dataset):

    def __init__(self, root, transform=None, size=(512, 512), return_meta=False):
        self.root = root
        self.resizor = torchvision.transforms.Resize(size)
        self.transform = transform
        self.return_meta = return_meta

        self.idxs = []
        self.objects = []

        self.get_data()

    def get_data(self):

        for i, img_name in enumerate(os.listdir(os.path.join(self.root, "imgs"))):
            img_path = os.path.join(self.root, "imgs", img_name)
            img_obj = tio.ScalarImage(img_path)

            seg_name = "{}.seg.nrrd".format(img_name.split(".")[0])
            seg_path = os.path.join(self.root, "interpolations2", seg_name)
            label_obj = tio.LabelMap(seg_path)
            labelmap = label_obj.tensor.squeeze(0)

            z_cords = torch.unique(torch.where(labelmap == True)[2])
            for z_cord in z_cords:
                self.idxs.append((i, z_cord))

            self.objects.append((img_obj, label_obj, img_name))



    def __len__(self):
        return len(self.idxs)

    def get_image_label_pair(self, img_obj, label_obj, z_cord):
        labelmap = label_obj.tensor.to(torch.uint8)
        label_slice = labelmap[:, :, :, z_cord]
        label_slice = self.resizor(label_slice)

        img = img_obj.tensor.to(torch.float)
        img_slice = img[:, :, :, z_cord]
        img_slice = self.resizor(img_slice)

        if self.transform:
            transformed = self.transform({'scan': img_slice.unsqueeze(-1),
                                          'annot': label_slice.unsqueeze(-1)})
            img_slice = transformed['scan'].squeeze(3).to(img_slice.dtype)
            label_slice = transformed['annot'].squeeze(3).to(label_slice.dtype)

        return img_slice, label_slice

    def __getitem__(self, idx):
        obj_idx, z_cord = self.idxs[idx]

        img_obj, label_obj, img_name = self.objects[obj_idx]

        img_slice, label_slice = self.get_image_label_pair(img_obj, label_obj, z_cord)
        if self.return_meta:
            return img_slice, label_slice, torch.empty((0, 0)), {"scan_name": img_name.split(".")[0]}
        else:
            return img_slice, label_slice, torch.empty((0, 0))


class AortaDatasetContextual(AortaDatasetInterpolation):

    def __getitem__(self, idx):
        obj_idx, z_cord = self.idxs[idx]

        img_obj, label_obj, img_name = self.objects[obj_idx]

        img_slice_current, label_slice_current = self.get_image_label_pair(img_obj, label_obj, z_cord)

        img_slice_prev, label_slice_prev = self.get_image_label_pair(img_obj, label_obj, z_cord + 1)
        contextual = torch.cat([img_slice_prev, label_slice_prev], 0)

        if self.return_meta:
            return img_slice_current, label_slice_current, contextual, {"scan_name": img_name.split(".")[0]}
        else:
            return img_slice_current, label_slice_current, contextual

    def get_image_label_pair(self, img_obj, label_obj, z_cord):
        labelmap = label_obj.tensor.to(torch.uint8)
        label_slice = labelmap[:, :, :, z_cord]
        label_slice = self.resizor(label_slice)

        img = img_obj.tensor.to(torch.float)
        img_slice = img[:, :, :, z_cord]
        img_slice = self.resizor(img_slice)

        if self.transform:
            transformed = self.transform({'scan': img_slice.unsqueeze(-1),
                                          'annot': label_slice.unsqueeze(-1)})
            img_slice = transformed['scan'].squeeze(3).to(img_slice.dtype)
            label_slice = transformed['annot'].squeeze(3).to(label_slice.dtype)

        return img_slice, label_slice

class AortaDatasetContextualTest(AortaDatasetInterpolation):
    def __init__(self, root, transform=None,  size=(512, 512), return_meta=False):
        label_path = os.path.join(root, "labels")
        image_path = os.path.join(root, "imgs")

        self.idxs = []
        self.objects = []
        self.root = root

        self.resizor = torchvision.transforms.Resize(size)

        self.transform = transform
        self.return_meta = return_meta
        self.label_files = sorted(os.listdir(label_path))
        self.label_list = [os.path.join(label_path, file) for file in self.label_files]
        self.image_files = sorted(os.listdir(image_path))
        # self.image_list = [os.path.join(image_path, file) for file in self.image_files]
        self.image_list = [os.path.join(image_path, file) for file in self.image_files if
                           os.path.splitext(file)[0] in [os.path.splitext(os.path.splitext(i)[0])[0] for i in
                                                         self.label_files]]
        self.get_data()
    def __getitem__(self, idx):

        obj_idx, z_cord = self.idxs[idx]

        img_obj, label_obj, img_name = self.objects[obj_idx]

        img_slice_current, label_slice_current = self.get_image_label_pair(img_obj, label_obj, z_cord)

        img_slice_prev, label_slice_prev = self.get_image_label_pair(img_obj, label_obj, z_cord + 1)
        contextual = torch.cat([img_slice_prev, label_slice_prev], 0)

        if self.return_meta:
            return img_slice_current, label_slice_current, contextual, {"scan_name": img_name.split(".")[0]}
        else:
            return img_slice_current, label_slice_current, contextual

    def get_image_dix(self, idx):
        a = [i.split(".")[0] for i in self.image_files]
        b = [i.split("%")[0][:7] for i in self.label_files]

        return a.index(b[idx])

    def get_data(self):

        for i, img_name in enumerate(self.label_list):

            label_map = self.label_list[i]
            label_obj = tio.LabelMap(label_map)
            lab = label_obj.tensor.squeeze(0)
            # find slice with annotation (2D)
            slice_idx = lab.max(dim=2)[1].max()
            # select label slice (2D)
            label = torch.select(lab, dim=2, index=slice_idx)

            img_idx = self.get_image_dix(i)
            img_path = self.image_list[img_idx]
            img_obj = tio.ScalarImage(img_path)

            z_cord = slice_idx

            img_name = str(img_obj.path)[-11:-4]

            self.idxs.append((i, z_cord))
            # z_cords = torch.unique(torch.where(labelmap == True)[2])
            # for z_cord in z_cords:
            #     self.idxs.append((i, z_cord))

            self.objects.append((img_obj, label_obj, img_name))

class DummySet(Dataset):

    def __len__(self):
        return 10

    def __getitem__(self, item):
        img = torch.randn(size=(1, 512, 512))
        label = torch.zeros(size=(512, 512)).type(torch.LongTensor)
        return img, label, None


def build_transforms(task):
    label_keys = ['annot']
    include = ['scan', 'annot']

    # resizor = T.Compose([
    #     T.Resize((100, 100, 1), include=include, label_keys=label_keys)
    # ])
    transforms_basic = T.Compose([
        T.Clamp(out_min=-100, out_max=500, include=include, label_keys=label_keys),
        T.preprocessing.RescaleIntensity((0, 1), include=include, label_keys=label_keys)
        # T.ZNormalization(include=include, label_keys=label_keys)
    ])

    if task == "train":
        transforms = T.Compose([
            transforms_basic
            # T.preprocessing.ZNormalization(include=include, label_keys=label_keys),
            # T.RandomFlip(axes=(0, 1, 2), include=include, label_keys=label_keys),
            # T.RandomAffine(include=include, label_keys=label_keys),
        ])
        return transforms
    elif task in ["test", "val", "infer"]:
        transforms = T.Compose([
            transforms_basic
            # T.preprocessing.ZNormalization(include=include, label_keys=label_keys)
        ])
        return transforms

    else:
        raise ValueError("task {} not supported".format(task))


# if __name__ == "__main__":
#
#     def plot(input, output):
#         plt.imshow(input.permute(1, 2, 0), cmap="gray")
#         plt.imshow(output, alpha=0.5, cmap='Reds')
#         plt.show()
#
#     image_path = r"E:\DTU_Aorta\specialkursus\data\train\imgs"
#     files = os.listdir(image_path)
#     label_path = r"E:\DTU_Aorta\specialkursus\data\train\labels"
#
#     dataset = AortaDataset(image_path, label_path, transform=build_transforms("train"))
#     dataloader = DataLoader(dataset, shuffle=True)
#     for (img, label) in dataloader:
#         bs = img.shape[0]
#         for i in range(bs):
#             plot(img[i], label[i].squeeze(0))


#
if __name__ == "__main__":
    pass
    data_root = r"D:\DTUTeams\wall_segmentation2022\specialkursus\data\test"

    a = AortaDatasetContextualTest(data_root, return_meta=True)
    b = next(iter(a))
    print(b)
    i = 0
    for img, label, _, name in a:
        # print(img.shape)
        # print(label.shape)
        print(name)
        i += 1
    print(i)
    # a[5]
    # for i in range(len(a)):
    #     img,label = a[i]
    #     if 2 in label:
    #         print(a.label_files[i])
    # print(img.dtype)
    # print(label.dtype)
    # root = r"E:\DTU_Aorta\specialkursus\data\train"
    # aortaDataset2 = AortaDatasetInterpolation(root)
    # aortaDataset2[5]
    # aortaDataset3 = AortaDatasetContextual(root)
    # aortaDataset3[5]
