# implementation partly taken from https://github.com/mseitzer/pytorch-fid

import numpy as np
import torch
from tqdm import tqdm
from scipy.linalg import sqrtm
from data import load_dataloader_preprocess
from inception_model import InceptionV3


def calculate_frechet_distance(mu_1, mu_2, cov_1, cov_2):
    mu_diff = np.sum((mu_1 - mu_2) ** 2)
    cov_mean = sqrtm(cov_1.dot(cov_2), disp=False)[0]
    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real
    trace = np.trace(cov_1 + cov_2 - 2 * cov_mean)
    return mu_diff + trace


def get_activation(dataloader, device, bs):
    model = InceptionV3().to(device)
    model.eval()
    pred_arr = np.empty((len(dataloader)*bs, 2048))

    start_idx = 0

    for batch, _ in tqdm(dataloader):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)[0]
        pred = pred.squeeze(3).squeeze(2).cpu().numpy()
        pred_arr[start_idx:start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return pred_arr

# the main function to calculate metric
# real_images, fake images - paths to folders containing images
def fid_metric(real_images, fake_images, device='cuda', image_size=64, bs=64):

    images_1 = load_dataloader_preprocess(image_size=image_size, bs=bs, path=real_images)
    images_2 = load_dataloader_preprocess(image_size=image_size, bs=bs, path=fake_images)

    act_1 = get_activation(images_1, device=device, bs=bs)
    act_2 = get_activation(images_2, device=device, bs=bs)

    mu_1, cov_1 = act_1.mean(axis=0), np.cov(act_1, rowvar=False)
    mu_2, cov_2 = act_2.mean(axis=0), np.cov(act_2, rowvar=False)

    return calculate_frechet_distance(mu_1, mu_2, cov_1, cov_2)
