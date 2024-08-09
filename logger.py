import os
from datetime import datetime
import socket
import torch
import sys
from torch.utils.tensorboard import SummaryWriter
import csv
import pickle
import pandas as pd

class Logger(object):
    def __init__(self, directory, write, save_freq=10, comment=""):
        self.parent_dir = directory
        self.write = write
        self.dir = directory
        self.fig_index = 0
        self.model_index = 0
        self.save_freq = save_freq
        if self.write:
            self.boardWriter = SummaryWriter(comment=comment)
            self.dir = self.boardWriter.log_dir
            self.log(f"Logs from {self.dir}\n{' '.join(sys.argv)}\n")
            self.df = pd.DataFrame(columns=['epoch', 'task', 'loss average'])

        self.save_dir = self.create_date()

    def create_date(self):
        e = datetime.now()
        date = '{}_{}_{}_{}'.format(e.day, e.month, e.year, str(e.hour)+str(e.minute)+str(e.second))
        os.makedirs(os.path.join(self.parent_dir, date))
        save_dir = os.path.join(self.parent_dir, date)

        return save_dir

    def write_to_board(self, name, scalars, index=0):
        self.log(f"{name} at {index}: {str(scalars)}")
        if self.write:
            for key, value in scalars.items():
                self.boardWriter.add_scalar(f"{name}/{key}", value, index)

    def plot_res(self, losses, distances):
        if len(losses) == 0 or not self.write:
            return
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(2)
        axs[0].plot(list(range(len(losses))), losses, color='orange')
        axs[0].set_xlabel("Steps")
        axs[0].set_ylabel("Loss")
        axs[0].set_title("Training")
        axs[0].set_yscale('log')
        for dist in distances:
            axs[1].plot(list(range(len(dist))), dist)
        axs[1].set_xlabel("Steps")
        axs[1].set_ylabel("Distance change")
        axs[1].set_title("Training")

        if self.fig_index > 0:
            os.remove(os.path.join(self.dir, f"res{self.fig_index-1}.png"))
        fig.savefig(os.path.join(self.dir, f"res{self.fig_index}.png"))
        self.boardWriter.add_figure(f"res{self.fig_index}", fig)
        self.fig_index += 1

    def log(self, message, step=0):
        # print(str(message))
        if self.write:
            # self.boardWriter.add_text("log", str(message), step)
            with open(os.path.join(self.dir, "logs.txt"), "a") as logs:
                logs.write(str(message) + "\n")

    def save_model(self, state_dict, name="dqn.pt", episode=1, forced=False):
        # write needs to be on in order to save model!!!
        if not self.write:
            return
        if (forced or
           (self.model_index > 0 and self.model_index % self.save_freq == 0)):
            torch.save({"model_state_dict": state_dict,
                        "episode": episode}, os.path.join(self.dir, name), pickle_protocol=4)

    def write_locations(self, row):
        self.log(str(row))
        if self.write:
            with open(os.path.join(self.dir, 'results.csv'),
                      mode='a', newline='') as f:
                res_writer = csv.writer(f)
                res_writer.writerow(row)

    def write_csv(self, epoch, task, loss_average):
        df_temp = pd.DataFrame(data={'epoch': [epoch],
                                     'task': [task],
                                     'loss average': [loss_average]})

        self.df = pd.concat([self.df, df_temp], ignore_index=True, axis=0)
        self.df.to_csv(os.path.join(self.save_dir, "results.csv"), index=False)


