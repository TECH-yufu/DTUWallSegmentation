import torch
import torch.nn.functional as F
from tqdm import tqdm
from criterion import dice_coeff, multiclass_dice_coeff
from hausdorff import hausdorff_distance

def hausdorffDistance(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    hausdorff_dist = 0

    for batch in tqdm(dataloader):
        image, mask_true, _ = batch
        image, mask_true = image.to(device), mask_true.to(device)
        mask_predicted = net(image).argmax(dim=1).unsqueeze(0)

        _,_,x1,y1 = torch.where(mask_true == 1)
        gt_set = torch.stack([x1,y1], dim=1)

        _,_,x2,y2 = torch.where(mask_predicted == 1)
        predicted_set = torch.stack([x2,y2], dim=1)

        hausdorff_dist += hausdorff_distance(gt_set.detach().cpu().numpy(), predicted_set.detach().cpu().numpy(), distance='manhattan')

    if num_val_batches == 0:
        return hausdorff_dist
    return hausdorff_dist / num_val_batches

def evaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    for batch in tqdm(dataloader):
        image, mask_true, _ = batch
        image = image.to(device)
        mask_true = mask_true.to(device)
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long) if len(mask_true.shape) == 3 else mask_true.to(dtype=torch.long).squeeze(1)
        mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)

            # convert to one-hot format
            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...],
                                                    reduce_batch_first=False) # background is class 0, so we use
                                                                              # class 1 (aorta wall) to compute the dice score

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score
    return dice_score / num_val_batches