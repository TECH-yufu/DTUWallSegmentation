import torchio as tio
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

# root = r"E:\DTU_Aorta\specialkursus\data\train\labels"
root = r'D:\DTUTeams\wall_segmentation2022\specialkursus\data\train\labels'
paths = [os.path.join(root, "DTU_076.seg.nrrd")] + [os.path.join(root, "DTU_076%{}.seg.nrrd".format(i)) for i in range(1, 6)]
n = 250

def get_edge_pixels(M,N, mode, img, l):

    if mode == 0:
        for m in range(M):
            for n in range(N-1,-1,-1):
                if img[n,m].item() == 1:
                    l.append((n,m))
                    break

    elif mode == 1:
        for m in range(M-1, -1, -1):
            for n in range(N-1,-1,-1):
                if img[m,n].item() == 1:
                    l.append((m,n))
                    break
    elif mode == 2:
        for m in range(M-1,-1,-1):
                for n in range(N):
                    if img[n,m].item() == 1:
                        l.append((n,m))
                        break
    elif mode == 3:
        for m in range(M):
            for n in range(N):
                if img[m,n].item() == 1:
                    l.append((m,n))
                    break
def unfold_image(img, center, max_dists=None, r_min=1, r_max=20, angles=30, steps=15):
    # Sampling angles and radii.
    angles = np.linspace(0, 2 * np.pi, angles, endpoint=False)
    distances = np.linspace(r_min, r_max, steps, endpoint=True)

    if max_dists is not None:
        max_dists.append(np.max(distances))

    # Get angles.
    angles_cos = np.cos(angles)
    angles_sin = np.sin(angles)

    # Calculate points positions.
    x_pos = center[0] + np.outer(angles_cos, distances)
    y_pos = center[1] + np.outer(angles_sin, distances)

    # Create list of sampling points.
    sampling_points = np.array([x_pos, y_pos]).transpose()
    sampling_shape = sampling_points.shape
    sampling_points_flat = sampling_points.reshape((-1, 2))

    # Sample from image.
    samples = map_coordinates(img, sampling_points_flat.transpose(), mode='nearest')
    samples = samples.reshape(sampling_shape[:2])

    return samples, sampling_points


def get_centre(edge_pixels):
    x = [i[0] for i in edge_pixels]
    y = [i[1] for i in edge_pixels]

    min_x = min(x)
    max_x = max(x)
    min_y = min(y)
    max_y = max(y)

    dx = (max_x - min_x)
    dy = (max_y - min_y)

    centre_x, centre_y = min_x + (dx) // 2, min_y + (dy) // 2

    return centre_x, centre_y
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




z_cords = []
circles_array = {0: [],
                 1: []}
for path in paths:
    labelmap =  tio.LabelMap(path).tensor.squeeze(0)



    z_cord = torch.unique(torch.where(labelmap == True)[2])
    z_cords.append(z_cord.item())
    image = labelmap[:, :, z_cord].squeeze(2)
    edges = filters.roberts(image.numpy())
    edges = edges != 0
    plt.imshow(edges)
    plt.show()

    edge_pixels = []
    M, N = edges.shape

    for i in range(4):
        get_edge_pixels(M, N, i, edges, edge_pixels)

    centre_x, centre_y = get_centre(edge_pixels)

    samples, sample_points = unfold_image(edges, center=(centre_x, centre_y), r_max=60, angles=1000, steps=1500)

    plt.figure(figsize=(15, 5))
    ax = plt.subplot(1, 3, 1, title='Sample positions in data')
    ax.imshow(edges, cmap='gray')
    ax.scatter(sample_points[..., 1], sample_points[..., 0], s=2, color='red')
    ax = plt.subplot(1, 3, 2, title='Sample positions and intensities')
    ax.scatter(sample_points[..., 1], sample_points[..., 0], c=samples, cmap='gray')
    ax = plt.subplot(1, 3, 3, title='Unfolded image')
    ax.imshow(samples, cmap='gray')
    plt.show()

    points_x, points_y = np.where(edges != 0)
    points = np.transpose(np.vstack([points_x, points_y]))







    # sc = cluster.SpectralClustering(n_clusters=2).fit(points)
    for cluster_label in [0,1]:
        # full_circle = points[np.where(sc.labels_ == cluster_label)]
        full_circle_points = points
        idxs = random.choices(np.arange(full_circle_points.shape[0]), k=n)
        trimmed_circle_points = full_circle_points[idxs, :]
        circle = Circle(trimmed_circle_points)
        circles_array[cluster_label].append(circle.sorted_points)

    # circle1 = points[np.where(sc.labels_ == 0)]
    # circle2 = points[np.where(sc.labels_ == 1)]

    # plt.scatter(circle1[:, 0], circle1[:, 1], label="circle1")
    # plt.scatter(circle2[:, 0], circle2[:, 1], label="circle2")
    # plt.legend()
    # plt.show()

    # circles = [circle1, circle2]
    # trimmed_circles = []
    # for circle in circles:
    #     n_points = circle.shape[0]
    #     idxs = random.choices(np.arange(n_points), k=n)
    #     trimmed_circle = circle[idxs, :]
    #     trimmed_circles.append(trimmed_circle)




    # plt.scatter(trimmed_circles[0][:, 0], trimmed_circles[0][:, 1], label="circle1")
    # plt.scatter(trimmed_circles[1][:, 0], trimmed_circles[1][:, 1], label="circle2")
    # plt.legend()
    # plt.show()
circles_array = {cluster_label: np.array(circles_array[cluster_label]) for cluster_label in [0, 1]}
z_cords = np.array(z_cords)
a = interp1d(z_cords, circles_array[0], axis=0)
images = []
for i in range(np.min(z_cords), np.max(z_cords), 10):
    interpolated_circle = a(i)
    fig = plt.figure()
    plt.scatter(interpolated_circle[:, 0], interpolated_circle[:, 1])
    plt.title("z = {}".format(i))
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    pil_object = Image.open(img_buf)
    images.append(pil_object)
    plt.close(fig=fig)
# images[0].save("E:/DTU_Aorta/specialkursus/inter.gif", save_all=True, append_images=images[1:], duration=50, loop=0)



