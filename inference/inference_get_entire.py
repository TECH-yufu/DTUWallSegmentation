import argparse
import torch.nn as nn
import os
import torchvision
from dataloader import build_transforms
import torchio as tio
import torch
from ContextualUnet import ContextualUnet
from unet import Unet
from tqdm import tqdm
import copy


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='weight path')
    parser.add_argument('--data_path', type=str, help='path to data')
    parser.add_argument('--save_root', type=str, help='path to save data')
    parser.add_argument('--img_size', help='Specifies the size of spatial dimensions', default=512, type=int)
    parser.add_argument('--contextual', help='If true, use contextual unet', action='store_true')
    return parser.parse_args()


def load_model(path, contextual):
    if contextual == True:
        model = ContextualUnet(input_c=1, output_c=2, context_c=2)
    else:
        model = Unet(input_c=1, output_c=2)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


class GetFullSegmentations:

    def __init__(self, model, data_path, save_root, img_size, contextual):
        self.model = model
        self.data_path = data_path
        self.save_root = save_root
        self.img_size = img_size
        self.resizor = torchvision.transforms.Resize((img_size, img_size))
        self.transforms = build_transforms('test')
        self.contextual = contextual

    def get_img_slice_prev(self, scan_tensor, z_cord):
        try:
            prev_slice = scan_tensor[:, :, :, z_cord]
            return prev_slice
        except IndexError:
            prev_slice = torch.zeros(scan_tensor.shape[:3], dtype=scan_tensor.dtype, device=device)
            return prev_slice

    def transform(self, scan_tensor):
        transformed = self.transforms({'scan': scan_tensor})
        scan_tensor = transformed['scan'].to(scan_tensor.dtype).to(device)
        return scan_tensor





    def get_segmentation(self, scan_obj):
        with torch.no_grad():
            scan_tensor = scan_obj.tensor.to(torch.float32)
            scan_tensor = self.transform(scan_tensor)
            scan_tensor = scan_tensor.to(device)
            inv_resizor = torchvision.transforms.Resize((scan_tensor.shape[1], scan_tensor.shape[2]))
            z_cords = torch.arange(scan_tensor.shape[3], device=device).flip([0])
            curr_output = torch.zeros((2, self.img_size, self.img_size), dtype=torch.float32, device=device)
            entire_output = torch.zeros((2, scan_tensor.shape[1], scan_tensor.shape[2], scan_tensor.shape[3]), dtype=torch.float32, device=device)
            for z_cord in tqdm(z_cords):
                img_slice = scan_tensor[:, :, :, z_cord]
                img_slice = self.resizor(img_slice)

                img_slice_prev = self.get_img_slice_prev(scan_tensor, z_cord + 1)
                img_slice_prev = self.resizor(img_slice_prev)
                contextual = torch.cat([img_slice_prev, curr_output.max(dim=0)[1].unsqueeze(0)], 0)

                curr_output = self.model(img_slice.unsqueeze(0), contextual.unsqueeze(0)).squeeze(0)

                entire_output[:, :, :, z_cord] = copy.deepcopy(inv_resizor(curr_output))
            segmentation = entire_output.max(dim=0)[1].unsqueeze(0).detach().cpu()
            return segmentation

    def main(self):
        self.model.eval()
        scan_root = os.path.join(self.data_path, "imgs")
        for scan_name in sorted(os.listdir(scan_root)):
            scan_path = os.path.join(scan_root, scan_name)
            scan_obj = tio.ScalarImage(scan_path)

            segmentation = self.get_segmentation(scan_obj)
            segmentation_obj = tio.LabelMap(tensor=segmentation, affine=scan_obj.affine)

            save_path = os.path.join(self.save_root, "{}.seg.nrrd".format(scan_name.split(".")[0]))
            segmentation_obj.save(save_path)





if __name__ == "__main__":
    args = parse()
    model = load_model(args.model_path, args.contextual)
    getFullSegmentations = GetFullSegmentations(model, args.data_path,
                                                args.save_root, args.img_size,
                                                args.contextual)
    getFullSegmentations.main()