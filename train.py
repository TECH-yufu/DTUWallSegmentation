import torch
import argparse
from dataloader import AortaDataset, DummySet, AortaDatasetInterpolation, build_transforms, AortaDatasetContextual
from torch.utils.data import DataLoader
from unet import Unet
from criterion import Criterion
from logger import Logger
from torch import optim
from tqdm import tqdm
from Evaluate import evaluate
from ContextualUnet import ContextualUnet
from sklearn.utils import class_weight

import os


class Trainer():

    def __init__(self, batch_size=4, img_size=512, epochs=1000, lambda_=1, lr=3e-4, dataloader_type='3d',
                 data_path=None, output_dir=None, xavier=False, contextual=True):
        self.batch_size = batch_size
        self.img_size = img_size
        self.epochs = epochs
        self.lambda_ = lambda_
        self.lr = lr
        self.dataloader_type = dataloader_type
        self.data_path = data_path
        self.contextual = contextual
        self.image_paths = {'train': r"{}/train/imgs".format(data_path),
                            'val': r"{}/val/imgs".format(data_path)}
        self.label_paths = {'train': r"{}/train/labels".format(data_path),
                            'val': r"{}/val/labels".format(data_path)}

        self.output_dir = output_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')

        # load the dataset
        self.dataloaders = self.loadDatasets(type=self.dataloader_type)

        # load the model
        if self.contextual:
            self.model = ContextualUnet(input_c=1, output_c=2, context_c=2, xavier=xavier)
        else:
            self.model = Unet(input_c=1, output_c=2, xavier=xavier)
        self.model.to(self.device)

        # load the criterion
        class_weights = self.get_class_weights(self.dataloaders['train'])
        self.criterion = Criterion(lambda_=self.lambda_, class_weights=class_weights.to(self.device))

        # load the optimizer
        # self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.lr, weight_decay=1e-8, momentum=0.9)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-8)
        # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', patience=2)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20)

        # load the logger
        self.logger = Logger(directory=self.output_dir, write=True)

    def get_class_weights(self, train_dataloader):
        class_weights = torch.tensor([1.0069, 145.4930])
        # class_weights = torch.tensor([1.0075, 135.1351])
        # class_weights = torch.tensor([899034194, 6673326])
        # class_weights = torch.tensor([0,0])
        #
        # for input, target, _ in train_dataloader:
        #     zeros = torch.count_nonzero(target == 0)
        #     ones = torch.numel(target) - zeros
        #     class_weights[0] += zeros
        #     class_weights[1] += ones
        #
        # class_weights = class_weights / class_weights.sum()
        # class_weights = 1/class_weights
        # print(class_weights)
        return class_weights

    def loadDatasets(self, type='3d'):
        assert type in ['3d', '2d', 'dummy']

        if self.contextual:
            datasets = {task: AortaDatasetContextual(os.path.join(self.data_path, task),
                                                        transform=build_transforms(task),
                                                        size=(self.img_size, self.img_size)) for task in
                        ['train', 'val']}
        else:
            if type == 'dummy':
                print("Using the dummy dataset!")
                datasets = {task: DummySet() for task in ['train', 'val']}
            elif type == '3d':
                datasets = {task: AortaDatasetInterpolation(os.path.join(self.data_path, task),
                                                            transform=build_transforms(task),
                                                            size=(self.img_size, self.img_size)) for task in
                            ['train', 'val']}
            elif type == '2d':
                datasets = {task: AortaDataset(os.path.join(self.data_path, task), transform=build_transforms(task),
                                               size=(self.img_size, self.img_size)) for task in ['train', 'val']}

        dataloaders = {task: DataLoader(datasets[task], batch_size=self.batch_size,
                                        shuffle=True, num_workers=0) for task in ['train', 'val']}

        return dataloaders

    def train(self):
        print("Training using:", self.device)

        best_val = 10 ** 9
        for i in (range(1, self.epochs + 1)):
            # every 10 epochs,
            if i % 10 == 0:
                task = 'val'
                self.model.eval()
            else:
                task = 'train'
                self.model.train()
            loss_avg = 0
            for input, target, con in tqdm(self.dataloaders[task]):
                input = input.to(self.device, dtype=torch.float)
                target = target.to(self.device)
                if con != None:
                    con = con.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(input, con)

                loss = self.criterion(output, target)

                if task == 'train':
                    loss.backward()
                    self.optimizer.step()

                loss_avg += loss.item()
            loss_avg = loss_avg / len(self.dataloaders[task])

            if task == 'train':
                # val_dice_score = evaluate(self.model, self.dataloaders['val'], self.device)
                # self.scheduler.step(val_dice_score)
                self.scheduler.step()

            elif task == 'val':
                if loss_avg < best_val:
                    best_val = loss_avg
                    self.logger.save_model(self.model.state_dict(), name="latest_unet.pt", episode=i, forced=True)
                    self.logger.save_model(self.model.state_dict(), name="best_unet.pt", episode=i, forced=True)
                    self.logger.log("Saving best unet model!")
                else:
                    self.logger.save_model(self.model.state_dict(), name="latest_unet.pt", episode=i, forced=True)

            with torch.no_grad():
                self.logger.write_csv(epoch=i, task=task, loss_average=loss_avg)
                self.logger.write_to_board(name=task, scalars={"loss_average": loss_avg}, index=i)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--batch_size', help='Size of one batch', default=4, type=int)
    parser.add_argument('--img_size', help='Specifies the size of spatial dimensions', default=512, type=int)
    parser.add_argument('--contextual', help='If true, use contextual unet', action='store_true')

    parser.add_argument('--epoch', help='Number of epochs to train for', default=int(1e3), type=int)

    parser.add_argument('--lambda_', help='Weighting of Dice score in the loss', default=1.0, type=float)

    parser.add_argument('--lr', help='Starting learning rate', default=3e-3, type=float)

    parser.add_argument('--dataloader_type', help="select one of ('2d', '3d' 'dummy')", type=str, default='3d')
    parser.add_argument('--xavier', help='If true, use the dummy dataset', action='store_true')

    parser.set_defaults(write=False)

    parser.add_argument(
        '--data_path', type=str,
        help="""Datapath to the train, val, and test sets. Each folder should contain a folder with images and labels.""")

    parser.add_argument(
        '--output_dir', type=str,
        help="""Output directory where the results are stored.""",
        default=r"D:\DTUTeams\wall_segmentation2022\specialkursus\github\DTUWallSegmentation\results")

    args = parser.parse_args()

    trainer = Trainer(batch_size=args.batch_size,
                      img_size=args.img_size,
                      epochs=args.epoch,
                      lambda_=args.lambda_,
                      lr=args.lr,
                      dataloader_type=args.dataloader_type,
                      data_path=args.data_path,
                      output_dir=args.output_dir,
                      xavier=args.xavier,
                      contextual=args.contextual)

    trainer.train()
