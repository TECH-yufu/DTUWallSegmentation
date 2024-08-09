import torch.nn as nn
import torch
from inference.inference import load_unet as load_regular
from inference.inference_contextual import load_unet as load_contextual
from torch.nn import Module
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
import argparse
from dataloader import AortaDataset, build_transforms, AortaDatasetContextual, AortaDatasetContextualTest
from criterion import dice_coeff, multiclass_dice_coeff
from tqdm import tqdm
import torch.nn.functional as F
import pandas as pd
from hausdorff import hausdorff_distance

from unet import Unet

def evaluate(net, dataloader, device, save_path):
    net.eval()
    data = {"scan_name": [], "dice_score": [],
            "haus_score": []}

    # iterate over the validation set
    for batch in tqdm(dataloader):
        imgs, labels, con, metadata = batch
        bs = imgs.shape[0]
        imgs = imgs.to(device)
        con = con.to(device)
        labels = labels.to(device)
        # move images and labels to correct device and type
        imgs = imgs.to(device=device, dtype=torch.float32)
        labels = labels.to(device=device, dtype=torch.long) if len(labels.shape) == 3 else labels.to(dtype=torch.long).squeeze(1)
        # mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            outputs = net(imgs, con)

            for i in range(bs):
                img, label, output, scan_name = imgs[i], labels[i], outputs[i], metadata['scan_name'][i]
                dice_score = dice_for_sample(net.n_classes, label, output)
                haus_score = hausdorf_for_sample(net.n_classes, label, output)

                data["scan_name"].append(scan_name)
                data['dice_score'].append(dice_score.item())
                data['haus_score'].append(haus_score)

    df = pd.DataFrame(data=data)
    df.to_csv(save_path, index=False)






    # net.train()
    #
    # # Fixes a potential division by zero error
    # if num_val_batches == 0:
    #     return dice_score
    # return dice_score / num_val_batches

def dice_for_sample(n_classes, label, output):
    # convert to one-hot format
    label_onehot = F.one_hot(label, n_classes).permute(2, 0, 1).float().unsqueeze(0)
    output_with_batch = output.unsqueeze(0)
    if n_classes == 1:
        mask_pred = (F.sigmoid(output_with_batch) > 0.5).float()
        # compute the Dice score
        dice_score = dice_coeff(mask_pred, label_onehot, reduce_batch_first=False)
    else:
        mask_pred = F.one_hot(output_with_batch.argmax(dim=1), n_classes).permute(0, 3, 1, 2).float()
        # compute the Dice score, ignoring background
        dice_score = multiclass_dice_coeff(mask_pred[:, 1:, ...], label_onehot[:, 1:, ...],
                                            reduce_batch_first=False)  # background is class 0, so we use
        # class 1 (aorta wall) to compute the dice score
    return dice_score

def hausdorf_for_sample(n_classes, label, output):
    mask_true = label.unsqueeze(0).unsqueeze(0)
    batched_output = output.unsqueeze(0)
    mask_predicted = batched_output.argmax(dim=1).unsqueeze(0)

    _, _, x1, y1 = torch.where(mask_true == 1)
    gt_set = torch.stack([x1, y1], dim=1)

    _, _, x2, y2 = torch.where(mask_predicted == 1)
    predicted_set = torch.stack([x2, y2], dim=1)

    hausdorff_dist = hausdorff_distance(gt_set.detach().cpu().numpy(), predicted_set.detach().cpu().numpy(),
                                         distance='manhattan')
    return hausdorff_dist



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='weight path')
    parser.add_argument('--data_path', type=str, help='path to images')
    parser.add_argument('--save_path', type=str, help='path to save csv')
    parser.add_argument('--batch-size', type=int, default=4, help='batch size')
    parser.add_argument('--contextual', help='If true, use contextual unet', action='store_true')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse()

    if args.contextual:
        unet = load_contextual(args.model_path)
        dataset = AortaDatasetContextualTest(args.data_path, transform=build_transforms('test'), return_meta=True)
    else:
        unet = load_regular(args.model_path)
        dataset = AortaDataset(args.data_path, transform=build_transforms("test"), return_meta=True)
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False)
    avg_dice = evaluate(unet, dataloader, device, args.save_path)
    print(avg_dice)

