import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torchvision.transforms import *

from src.training import train
from src.utils import get_logger, set_seed, Cutout


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--epochs", required=True, type=int)
    parser.add_argument("--lr", required=True, type=float, help="learning rate")
    parser.add_argument("--batch-size", default=128, type=int, help="Path to val images directory")
    parser.add_argument("--crop-size", default=224, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--train-data", default="data/imagewoof2/train", help="Path to train images directory")
    parser.add_argument("--val-data", default="data/imagewoof2/val", help="Path to val images directory")
    parser.add_argument("--model-save-dir", default='pretrained')
    return parser.parse_args()


class HeadE(nn.Module):
    def __init__(self, input_size, output_size, dropout_p):
        super(HeadE, self).__init__()
        self.bn = nn.BatchNorm1d(input_size)
        self.drop = nn.Dropout(dropout_p, inplace=True)
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        x = self.bn(input)
        x = self.drop(x)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    args = parse_arguments()

    set_seed(42)

    normalize_transform = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    test_transforms = Compose([Resize(args.crop_size), CenterCrop(args.crop_size),
                              ToTensor(), normalize_transform])

    train_transforms = Compose([RandomApply([RandomRotation(15, fill=(127,127,127))], p=0.8),
                                RandomAffine(degrees=0, scale=(0.83, 1.2), translate=(0.1, 0.1), fillcolor=(127,127,127)),
                                Resize(args.crop_size), CenterCrop(args.crop_size),
                                RandomHorizontalFlip(p=0.5),
                                ToTensor(), 
                                RandomApply([Cutout(size=50, color=(0.5,0.5,0.5))], p=0.5),
                                normalize_transform])

    val_dataset = ImageFolder(args.val_data, test_transforms)

    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=2*args.batch_size, 
                                                  num_workers=2, pin_memory=True,
                                                  shuffle=False, drop_last=False)

    train_dataset = ImageFolder(args.train_data, train_transforms)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, 
                                                  num_workers=2, pin_memory=True,
                                                  shuffle=True, drop_last=True)

    model = torchvision.models.resnet50(pretrained=(not args.resume))
    model.fc = HeadE(model.fc.in_features, len(train_dataset.classes), dropout_p=0.5)

    start_epoch, best_val_acc = 1, 0
    learning_rate = args.lr
    device = torch.device('cuda:0' if args.gpu else 'cpu')

    model_path = os.path.join(args.model_save_dir, args.model_name + '.pth')

    if args.resume:
        if os.path.isfile(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epochs'] + 1
            best_val_acc = checkpoint['accuracy']
            print("loaded checkpoint '{}' (epochs: {}, accuracy: {})"
                  .format(model_path, start_epoch - 1, best_val_acc))
        else:
            sys.exit("No checkpoint found at '{}' to resume training.".format(model_path))
    elif os.path.isfile(model_path):
        sys.exit("Checkpoint '{}' already exists. Name model differently or set the '--resume' flag.".format(model_path))
    
    model.to(device)

    loss_fn = F.cross_entropy
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=2e-4)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=3)

    log_path = os.path.join(args.model_save_dir, args.model_name + '.log')
    logger = get_logger(log_path)

    train(model, train_dataloader, val_dataloader, loss_fn, optimizer, 
          args.epochs, device, logger, model_path, start_epoch=start_epoch, best_val_acc=best_val_acc)
