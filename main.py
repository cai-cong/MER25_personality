# -*- coding: utf-8 -*-
import argparse
import torch.nn as nn
from dataset import get_dataloader
from train import train_model
from model import Model
import os
from dateutil import tz
from datetime import datetime
import sys
from dataset_load import load_data


class Logger(object):
    def __init__(self, log_file="log_file.log"):
        self.terminal = sys.stdout
        self.file = open(log_file, "w")

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.file.flush()

def main():

    parser = argparse.ArgumentParser()


    parser.add_argument('--dataset_file_path', default="/root/private_data/mdpe/deception/features/")
    parser.add_argument('--data_source', default="deception", help='emotion or deception folder') 
    parser.add_argument('--feature_set',nargs='+',default="baichuan13B-base", type=str, help='Specify the features (one or several).')

    parser.add_argument('--label_personality', default="/root/private_data/mdpe/label_personality_normalized3.csv", help='label_personality csv path')
    parser.add_argument('--partition', default="/root/private_data/mdpe/partition3.csv", help='partition csv path')

    parser.add_argument('--classnum', default=5, type=int)
    parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument('--dropout', default=0.5, type=float)

    parser.add_argument('--epochs', default=50, type=int, help='number of training epochs')
    parser.add_argument('--batch_size', default=64, type=int, help='size of a mini-batch')
    parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
    parser.add_argument('--parallel', action='store_true',help='whether use DataParallel')

    args = parser.parse_args()

    log_path = "./log/"
    os.makedirs(log_path, exist_ok=True)
    args.log_file_name = '{}_[{}_{}]_[{}_{}_{}_{}]'.format(
        datetime.now(tz=tz.gettz()).strftime("%Y-%m-%d-%H-%M-%S"),args.data_source, args.feature_set, args.lr, args.hidden_size, args.dropout, args.batch_size)

    sys.stdout = Logger(os.path.join(log_path, args.log_file_name + '.log'))
    print(f"Log file name (modelname): {args.log_file_name}")

    data = load_data(args)

    train_dataloader, dev_dataloader = get_dataloader(args,data)

    args.fea_dim = train_dataloader.dataset .get_feature_dim()
    model = Model(args)

    if args.parallel:
        model = nn.DataParallel(model)

    model = model.cuda()
    print('=' * 50)

    train_model(model, train_dataloader,dev_dataloader, args.epochs, args.lr, args.log_file_name)

    print('=' * 50)

if __name__ == "__main__":
    main()
