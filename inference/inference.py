import sys

sys.path.append("..")

import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import argparse
from dataloader import AortaDataset, build_transforms, DummySet, AortaDatasetInterpolation

from unet import Unet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Inference(nn.Module):

    def __init__(self, unet: nn.Module):
        super().__init__()
        self.unet = unet

    def forward(self, x):
        x = self.unet(x)
        x = -(x[:, 0, :, :] - x[:, 1, :, :])
        # x = torch.abs(x)
        # x = x[:, 0, :, :]
        return x


class Inference2(nn.Module):
    def __init__(self, unet: nn.Module):
        super().__init__()
        self.unet = unet

    def forward(self, x):
        x = self.unet(x)
        x = x.max(dim=1)[1]
        # x = torch.abs(x)
        # x = x[:, 0, :, :]
        return x


def plot(input, output, label):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(input.permute(1, 2, 0), cmap="gray")
    ax1.imshow(output, alpha=0.7, cmap='Reds')
    ax1.title.set_text("Model's output")
    ax2.imshow(input.permute(1, 2, 0), cmap="gray")
    ax2.imshow(label, alpha=0.5, cmap='Reds')
    ax2.title.set_text("Ground truth")
    plt.show()


def load_unet(path):
    unet = Unet(input_c=1, output_c=2)

    checkpoint = torch.load(path, map_location=device)
    unet.load_state_dict(checkpoint["model_state_dict"])
    unet.to(device)
    unet.eval()
    return unet


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='weight path')
    parser.add_argument('--data_path', type=str, help='path to data')
    parser.add_argument('--batch-size', type=int, default=4, help='batch size')
    parser.add_argument('--dataloader_type', help="select one of ('2d', '3d' 'dummy')", type=str, default='3d')
    parser.add_argument('--img_size', help='Specifies the size of spatial dimensions', default=512, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse()
    unet = load_unet(args.model_path)
    inference = Inference2(unet)
    assert args.dataloader_type in ['dummy', '3d', '2d']
    if args.dataloader_type == 'dummy':
        print("Using the dummy dataset!")
        dataset = DummySet()
    elif args.dataloader_type == '3d':
        dataset = AortaDatasetInterpolation(args.data_path,
                                            transform=build_transforms("test"),
                                            size=(args.img_size, args.img_size))
    elif args.dataloader_type == '2d':
        dataset = AortaDataset(args.data_path, transform=build_transforms('test'),
                               size=(args.img_size, args.img_size))
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)

    for input, label, _ in dataloader:
        input = input.to(device, dtype=torch.float)
        output = inference(input)

        bs = input.shape[0]
        for i in range(bs):
            plot(input[i].detach().cpu(), output[i].detach().cpu(), label[i].squeeze(0).detach().cpu())
