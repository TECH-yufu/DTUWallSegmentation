import torchio as tio
import os
import torch
import torchvision.transforms
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage


def get_edge_pixels(M,N, mode, img, l, l2):

    if mode == 0:
        for m in range(M):
            target = 1
            for n in range(N-1,-1,-1):
                if img[0,n,m].item() == 1 and target == 1:
                    l.append((n,m))
                    target = 0
                elif img[0,n,m].item() == 0 and target == 0:
                    l2.append((n+1,m))
                    break

    if mode == 1:
        for n in range(N-1,-1,-1):
            target = 1
            for m in range(M-1, -1, -1):
                if img[0,n,m].item() == 1 and target == 1:
                    l.append((n,m))
                    target = 0
                elif img[0,n,m].item() == 0 and target == 0:
                    l2.append((n,m+1))
                    break

    if mode == 2:
        for m in range(M-1,-1,-1):
            target = 1
            for n in range(N):
                if img[0,n,m].item() == 1 and target == 1:
                    l.append((n,m))
                    target = 0
                elif img[0,n,m].item() == 0 and target == 0:
                    l2.append((n-1,m))
                    break

    if mode == 3:
        for n in range(N):
            target = 1
            for m in range(M):
                if img[0,n,m].item() == 1 and target == 1:
                    l.append((n,m))
                    target = 0
                elif img[0,n,m].item() == 0 and target == 0:
                    l2.append((n,m-1))
                    break

def get_outer_and_inner(s_edge_pixels1, s_edge_pixels2):
    s1 = s_edge_pixels1
    s2 = s_edge_pixels2

    # intersection = s1.intersection(s2)
    outer = s1
    inner = s2 - s1

    return list(outer), list(inner)



path = r'D:\DTUTeams\wall_segmentation2022\specialkursus\data\train\labels'

for f in sorted(os.listdir(path)):
    file = os.path.join(path, f)

    resize = torchvision.transforms.Resize((512,512))

    label = tio.LabelMap(file)

    l = label.tensor

    # find slice with annotation (2D)
    slice_idx = l.max(dim=3)[1].max()
    # select label slice (2D)
    label = torch.select(l, dim=3, index=slice_idx)
    label = resize(label)

    edge_pixels1 = []
    edge_pixels2 = []
    _, M, N = label.shape

    for i in range(4):
        get_edge_pixels(M,N,i, label, edge_pixels1, edge_pixels2)

    s_edge_pixels1 = set(edge_pixels1)
    s_edge_pixels2 = set(edge_pixels2)

    outer, inner = get_outer_and_inner(s_edge_pixels1, s_edge_pixels2)

    label_edge = torch.zeros(512,512)
    for p in outer:
        x,y = p
        label_edge[x,y] = 1

    for p in inner:
        x,y = p
        label_edge[x,y] = 1


    label_edge_f = ndimage.median_filter(label_edge, size=2).astype(int)

    plt.imshow(label_edge_f)
    plt.show()