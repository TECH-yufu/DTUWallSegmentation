import torch
from dataloader import AortaDataset, DummySet
from unet import Unet
from torch.utils.data import DataLoader
from criterion import dice_coeff, multiclass_dice_coeff
from tqdm import tqdm
import torch.nn.functional as F

def evaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
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

batch_size = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

datasets = {task: DummySet() for task in ['train', 'val']}
dataloaders = {task: DataLoader(datasets[task], batch_size=batch_size,
                        shuffle=True) for task in ['train', 'val']}

# load the model
model = Unet(input_c=1, output_c=2)
model.to(device)

a = evaluate(model, dataloaders['val'], device)
print(a)

