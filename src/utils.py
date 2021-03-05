import logging
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid
from tqdm import tqdm


def get_logger(filename):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(message)s', '%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(filename)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


class Cutout:
    """Аугментация, которая вырезает из картинки квадрат со стороной 2*size"""
    def __init__(self, size=8, color=(0, 0, 0)):
        self.size = size
        self.color = color
        assert all(map(lambda x: 0 <= x <= 1, color)) 

    def __call__(self, tensor):
        _, h, w = tensor.shape
        y = np.random.randint(0, h)
        x = np.random.randint(0, w)
        r, g, b = self.color
        tensor[:, y - self.size:y + self.size, x - self.size:x + self.size] = \
                torch.FloatTensor([[[r]], [[g]], [[b]]])
        return tensor


def predict(model, loader, loss_fn, device):
    model.eval()

    predicts = []
    losses = []

    for batch in tqdm(loader, total=len(loader)):
        input = batch[0].to(device)
        target = batch[1].to(device)
        with torch.no_grad():
            output = model(input)
        pred_labels = torch.max(output, 1)[1]
        predicts.append(pred_labels)
        loss = loss_fn(output, target, reduction='none')
        losses.append(loss)
    
    predicts = torch.cat(predicts).tolist()
    losses = torch.cat(losses).tolist()

    return predicts, losses


def denormalize(images, mean, std):
    mean = torch.tensor(mean).reshape(1,3,1,1)
    std = torch.tensor(std).reshape(1,3,1,1)
    return images * std + mean


def show_batch(batch, mean=None, std=None, max_show=64, figsize=(12, 12)):
    images, labels = batch
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xticks([]), ax.set_yticks([])
    if mean is None or std is None:
        images_denormed = images[:max_show]
    else:
        images_denormed = denormalize(images[:max_show], mean, std)
    ax.imshow(make_grid(images_denormed).permute(1,2,0).clamp(0,1))


def show_top_k_misclassified(k, pred_labels, losses, test_dataset, mean, std, num_cols=5):
    top_k_misclassified_idx = np.argsort(losses)[-k:]

    true_labels = np.array(test_dataset.targets)[top_k_misclassified_idx]
    pred_labels = np.array(pred_labels)[top_k_misclassified_idx]

    num_rows = k // num_cols + int(k % num_cols != 0)
    plt.figure(figsize=(12, 2 * num_rows))

    for i, idx in enumerate(top_k_misclassified_idx):
        image = denormalize(test_dataset[idx][0], mean, std)[0].permute(1, 2, 0).clamp(0, 1)
        header = 'predicted label: {}\ntrue label: {}'.format(pred_labels[i], true_labels[i])
        plt.subplot(num_rows, num_cols, i+1)
        # plt.imshow(image[:, :, ::-1])
        plt.imshow(image)
        plt.title(header)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	os.environ['PYTHONHASHSEED'] = str(seed)
