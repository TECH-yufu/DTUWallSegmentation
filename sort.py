import os
import torchio as tio
import torch

class SorterFiler():

    def __init__(self, data_root):

        self.out_path = os.path.join(data_root, "labels_test")
        self.label_path = os.path.join(data_root, "labels")
        self.image_path = os.path.join(data_root, "imgs")

        self.label_files = sorted(os.listdir(self.label_path))
        self.label_list = [os.path.join(self.label_path, file) for file in self.label_files]
        self.image_files = sorted(os.listdir(self.image_path))
        # self.image_list = [os.path.join(image_path, file) for file in self.image_files]
        self.image_list = [os.path.join(self.image_path, file) for file in self.image_files if
                           os.path.splitext(file)[0] in [os.path.splitext(os.path.splitext(i)[0])[0] for i in
                                                         self.label_files]]

    def sorter(self):

        for image in self.image_list:
            name = image.split("\\")[-1].split(".")[0]

            list_to_sort = []

            for i in range(len(self.label_files)):
                label_map = self.label_list[i]
                if name in label_map:
                    label_obj = tio.LabelMap(label_map)
                    lab = label_obj.tensor
                    # find slice with annotation (2D)
                    z = torch.unique(torch.where(lab == 1)[3])

                    # get name
                    label_name = label_map.split("\\")[-1].split(".")[0]

                    list_to_sort.append((z, label_name))

            self.rename(list_to_sort)
            print(list_to_sort)


    def rename(self, list_to_sort):

        s = sorted(list_to_sort, key=lambda x: x[0])
        x = sorted(list_to_sort, key=lambda x: x[1], reverse=True)
        for i in range(len(s)):
            z_cord, label_name = s[i]

            os.rename(os.path.join(self.label_path, label_name+'.seg.nrrd'), os.path.join(self.out_path, x[i][1]+'.seg.nrrd'))


            # label_obj.save(os.path.join(self.out_path, x[i][1]+'.seg.nrrd'))

    def get_image_idx(self, idx):
        a = [i.split(".")[0] for i in self.image_files]
        b = [i.split("%")[0][:7] for i in self.label_files]

        return a.index(b[idx])

if __name__ == "__main__":
    data_root = r"D:\DTUTeams\wall_segmentation2022\specialkursus\data\test"
    a = SorterFiler(data_root)
    list_to_sort = a.sorter()
    print(a.image_list)

