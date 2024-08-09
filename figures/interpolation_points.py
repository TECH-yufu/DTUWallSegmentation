import torchio as tio
from scipy import ndimage
from skimage import filters
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import numpy as np
from scipy.interpolate import interp1d
from PIL import Image
import io
import math
from scipy.ndimage.interpolation import map_coordinates
import torchvision
from tqdm import tqdm
from skimage.draw import polygon
import torchio as tio



class Circle:

    def __init__(self, euc_points):
        self.euc_points = euc_points
        self.center = np.array([np.mean(euc_points[:, 0]), np.mean(euc_points[:, 1])])
        self.refvec = np.array([0,1])
        self.sorted_points = self.get_sorted()


    def clockwiseangle_and_distance(self, point):
        # Vector between point and the self.center: v = p - o
        vector = [point[0] - self.center[0], point[1] - self.center[1]]
        # Length of vector: ||v||
        lenvector = math.hypot(vector[0], vector[1])
        # If length is zero there is no angle
        if lenvector == 0:
            return -math.pi, 0
        # Normalize vector: v/||v||
        normalized = [vector[0] / lenvector, vector[1] / lenvector]
        dotprod = normalized[0] * self.refvec[0] + normalized[1] * self.refvec[1]  # x1*x2 + y1*y2
        diffprod = self.refvec[1] * normalized[0] - self.refvec[0] * normalized[1]  # x1*y2 - y1*x2
        angle = math.atan2(diffprod, dotprod)
        # Negative angles represent counter-clockwise angles so we need to subtract them
        # from 2*pi (360 degrees)
        if angle < 0:
            return 2 * math.pi + angle, lenvector
        # I return first the angle because that's the primary sorting criterium
        # but if two vectors have the same angle then the shorter distance should come first.
        return angle, lenvector

    def get_sorted(self):
        return sorted(self.euc_points, key=self.clockwiseangle_and_distance)



    def get_mask(self, size=(512, 512)):
        mask = torch.zeros(size=size, dtype=torch.bool)

        fit_idxs = np.arange(len(self.euc_points))
        interpolator = interp1d(fit_idxs, self.euc_points, axis=0)

        # plt.scatter(self.euc_points[:, 0], self.euc_points[:, 1])
        # plt.show()


        points = []
        interpolation_idxs = np.linspace(np.min(fit_idxs), np.max(fit_idxs), 2000)
        for interpolation_idx in interpolation_idxs:
            point = interpolator(interpolation_idx)
            points.append(point)

        points = np.vstack(points)
        # plt.scatter(points[:, 0], points[:, 1])
        # plt.show()


        rr, cc = polygon(points[:, 0], points[:, 1])
        rr = np.clip(rr, 0, size[0] - 1)
        cc = np.clip(cc, 0, size[1] - 1)
        mask[rr, cc] = 1
        return mask




def get_circles(labelmap_slice):
    def get_edge_pixels(M, N, mode, img, l, l2):

        if mode == 0:
            for m in range(M):
                target = 1
                for n in range(N - 1, -1, -1):
                    if img[0, n, m].item() == 1 and target == 1:
                        l.append((n, m))
                        target = 0
                    elif img[0, n, m].item() == 0 and target == 0:
                        l2.append((n + 1, m))
                        break

        if mode == 1:
            for n in range(N - 1, -1, -1):
                target = 1
                for m in range(M - 1, -1, -1):
                    if img[0, n, m].item() == 1 and target == 1:
                        l.append((n, m))
                        target = 0
                    elif img[0, n, m].item() == 0 and target == 0:
                        l2.append((n, m + 1))
                        break

        if mode == 2:
            for m in range(M - 1, -1, -1):
                target = 1
                for n in range(N):
                    if img[0, n, m].item() == 1 and target == 1:
                        l.append((n, m))
                        target = 0
                    elif img[0, n, m].item() == 0 and target == 0:
                        l2.append((n - 1, m))
                        break

        if mode == 3:
            for n in range(N):
                target = 1
                for m in range(M):
                    if img[0, n, m].item() == 1 and target == 1:
                        l.append((n, m))
                        target = 0
                    elif img[0, n, m].item() == 0 and target == 0:
                        l2.append((n, m - 1))
                        break

    def get_outer_and_inner(s_edge_pixels1, s_edge_pixels2):
        s1 = s_edge_pixels1
        s2 = s_edge_pixels2

        # intersection = s1.intersection(s2)
        outer = s1
        inner = s2 - s1

        return list(outer), list(inner)



    # resize = torchvision.transforms.Resize((512, 512))

    # labelmap_slice = resize(labelmap_slice.unsqueeze(0))
    labelmap_slice = labelmap_slice.unsqueeze(0)

    edge_pixels1 = []
    edge_pixels2 = []
    _, M, N = labelmap_slice.shape

    for i in range(4):
        get_edge_pixels(M, N, i, labelmap_slice, edge_pixels1, edge_pixels2)

    s_edge_pixels1 = set(edge_pixels1)
    s_edge_pixels2 = set(edge_pixels2)

    outer, inner = get_outer_and_inner(s_edge_pixels1, s_edge_pixels2)

    label_edge = torch.zeros(512, 512)
    label_edge2 = torch.zeros(512, 512)
    for p in outer:
        x, y = p
        label_edge[x, y] = 1

    for p in inner:
        x, y = p
        label_edge2[x, y] = 1

    label_edge_f = ndimage.median_filter(label_edge2, size=2).astype(int)

    # plt.imshow(label_edge_f)
    # plt.show()

    return label_edge, torch.tensor(label_edge_f)

def save_gif(z_cords, interpolation, save_path):
    images = []
    for i in range(np.min(z_cords), np.max(z_cords), 1):
        interpolated_circle_points = interpolation(i)

        circle = Circle(interpolated_circle_points)
        mask = circle.get_mask((512, 512))

        fig = plt.figure()
        # plt.scatter(interpolated_circle_points[:, 0], interpolated_circle_points[:, 1])
        plt.imshow(mask)
        plt.title("z = {}".format(i))
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        pil_object = Image.open(img_buf)
        images.append(pil_object)
        plt.close(fig=fig)
    images[0].save(save_path, save_all=True, append_images=images[1:], duration=12, loop=0)

def save_labelmap(z_cords, interpolators, native_size, affine, save_path):
    labelmap_tensor = torch.zeros(native_size)


    for i in range(np.min(z_cords), np.max(z_cords)):
        outer_points = interpolators['outer'](i)
        outer_circle = Circle(outer_points)
        outer_mask = outer_circle.get_mask( (native_size[0], native_size[1]) )

        inner_points = interpolators['inner'](i)
        inner_circle = Circle(inner_points)
        inner_mask = inner_circle.get_mask((native_size[0], native_size[1]))

        labelmap_slice = torch.logical_and(outer_mask, ~inner_mask)
        labelmap_tensor[:, :, i] = labelmap_slice

    labelmap_tio = tio.LabelMap(tensor=labelmap_tensor.unsqueeze(0), affine=affine)
    # labelmap_tio.save(save_path)






def main(paths, n, file_name, q):
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError()


    z_cords = []
    circles_array = {'inner': [],
                     'outer': []}
    native_sizes = []

    for path in paths:

        l = tio.LabelMap(path)

        labelmap = l.tensor.squeeze(0)
        affine = l.affine
        native_sizes.append(labelmap.shape)

        z_cord = torch.unique(torch.where(labelmap == True)[2])
        z_cords.append(z_cord.item())
        labelmap_slice = labelmap[:, :, z_cord].squeeze(2)

        outer, inner = get_circles(labelmap_slice)
        circles_img = {"inner": inner,
                       "outer": outer}

        ####rodekode skal fjernes ####
        img = tio.ScalarImage(r"E:\DTU_Aorta\specialkursus\data\val\imgs\DTU_053.nii").tensor.squeeze(0)

        # plt.imshow(outer, cmap='Reds')
        # plt.imshow(inner, cmap="Blues")
        plt.imshow(np.transpose(labelmap_slice), cmap="Reds")
        plt.imshow(np.transpose(img[:, :, z_cord], (1,0,2)), cmap='gray', alpha=0.7)
        plt.show()
        ####rodekode skal fjernes ####

        plt.imshow(np.transpose(img[:, :, z_cord][140:300, 140:300], (1, 0, 2)), cmap='gray', alpha=0.7)
        for label, circle_img in circles_img.items():
            points_x, points_y = torch.where(circle_img == 1)
            full_circle_points = torch.stack([points_x, points_y], dim=1)

            idxs = random.choices(np.arange(full_circle_points.shape[0]), k=n)
            trimmed_circle_points = full_circle_points[idxs, :]

            circle = Circle(trimmed_circle_points.numpy())


            # overwrite the points so that they are sorted clickwise
            circles_array[label].append(circle.sorted_points)


            # plot skal fjernes efter
            plt.scatter(trimmed_circle_points[:, 0]-140, trimmed_circle_points[:, 1]-140, label=label, s=[32]*trimmed_circle_points.shape[0])


        plt.legend(fontsize=25)
        plt.title("Band edges modelled as points", fontsize=25)
        plt.axis('off')
        plt.show()

    assert [native_sizes[0]]*len(native_sizes) == native_sizes

    # make array
    z_cords = np.array(z_cords)

    interpolators = {}

    for label in ['inner', 'outer']:
        # make array
        circle_array = np.array(circles_array[label])
        interpolator = interp1d(z_cords, circle_array, axis=0)
        interpolators[label] = interpolator

    save_labelmap(z_cords, interpolators, native_sizes[0], affine, save_path="E:/DTU_Aorta/specialkursus/data/{}/interpolations/{}.seg.nrrd".format(q, file_name))

        # save_gif(z_cords, interpolator, "E:/DTU_Aorta/specialkursus/{}.gif".format(label))





    # make array
    # circles_array = {cluster_label: np.array(circles_array[cluster_label]) for cluster_label in ['inner', 'outer']}
    # z_cords = np.array(z_cords)

    # make interpolation









if __name__ == "__main__":
    n = 75

    for q in ['val', 'test']:

        root = r"E:\DTU_Aorta\specialkursus\data\{}\labels".format(q)
        img_paths = r"E:\DTU_Aorta\specialkursus\data\{}\imgs".format(q)

        for f in tqdm(sorted(os.listdir(img_paths))):
            file_name = os.path.join(f.split(".")[0])
            paths = [os.path.join(root, file_name+".seg.nrrd")] + [os.path.join(root, file_name+"%{}.seg.nrrd".format(i)) for i in range(1, 6)]

            try:
                main(paths, n, file_name,q)
            except FileNotFoundError:
                print("{} mangler annoteringer".format(file_name))


