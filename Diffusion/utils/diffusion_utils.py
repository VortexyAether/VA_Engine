import torch
import torch.nn.functional as F
import numpy as np
import logging
import os
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

def make_beta_schedule(timesteps, s=0.008):
    timesteps = (torch.arange(timesteps + 1, dtype=torch.float64) / timesteps + s)
    alphas = timesteps / (1 + s) * np.pi / 2
    alphas = torch.cos(alphas).pow(2)
    alphas = alphas / alphas[0]
    betas = 1 - alphas[1:] / alphas[:-1]
    betas = np.clip(betas, a_min=0, a_max=0.999)

    return betas

def setup_logging(log_dir, log_file_name="training.log"):
    os.makedirs(log_dir, exist_ok=True)

    log_file_path = os.path.join(log_dir, log_file_name)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging initialized. Logs will be saved to {log_file_path}")