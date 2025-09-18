import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Subset

import SAM
from datasets_torch import *
from utils import (cal_recon_loss, plot_all_attack_results,
                   plot_attack_results, plotX, set_logging)


def parse_args():
    parser = argparse.ArgumentParser(
        description="FSHA Training with configurable alpha"
    )
    parser.add_argument("--alpha", type=float, default=10, help="Alpha parameter")
    parser.add_argument(
        "--lr_f", type=float, default=0.0001, help="Learning rate for f"
    )
    parser.add_argument(
        "--lr_tilde", type=float, default=0.00015, help="Learning rate for tilde_f"
    )
    parser.add_argument(
        "--lr_D", type=float, default=0.0001, help="Learning rate for D"
    )
    parser.add_argument(
        "--gradient_penalty", type=float, default=500, help="Gradient penalty"
    )
    parser.add_argument(
        "--iteration", type=int, default=10000, help="Number of training iterations"
    )
    parser.add_argument("--dataset", type=str, default="cifar", help="Dataset to use")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--id_setup", type=int, default=4, help="Model setup ID")
    parser.add_argument("--gpu", type=str, default="0", help="GPU device ID")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    return parser.parse_args()


args = parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "" + args.gpu
debug = args.debug
dataset = args.dataset
batch_size = args.batch_size
id_setup = args.id_setup

hparams = {
    "WGAN": True,
    "gradient_penalty": args.gradient_penalty,
    "style_loss": None,
    "lr_f": args.lr_f,
    "lr_tilde": args.lr_tilde,
    "lr_D": args.lr_D,
    "alpha": args.alpha,
}

iteration = args.iteration
log_frequency = 500


from datetime import datetime

current_time = datetime.now()
formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")

images_dir = f"./results/main_sam/{dataset}/{id_setup}/{formatted_time}"
if not os.path.exists(images_dir):
    os.makedirs(images_dir)

log_file_name = f"exp.log"
set_logging(images_dir, log_file_name)

xpriv, xpub = get_dataset(dataset)
num_classes = get_num_classes(xpriv)
reconstruct_eval_dataset = Subset(xpriv, range(500))

if debug:
    xpriv = Subset(xpriv, range(100))
    xpub = Subset(xpub, range(100))
    iteration = 10
    log_frequency = 5
else:
    pass


logging.info(f"hparams: {hparams}")

fsha = SAM.SAM(
    xpriv,
    xpub,
    id_setup - 1,
    batch_size,
    hparams,
    num_classes,
    reconstruct_eval_dataset,
)

LOG = fsha(iteration, verbose=True, progress_bar=True, log_frequency=log_frequency)

from SAM import results as splitguard_results
from utils import plot_splitguard_result

plot_splitguard_result(splitguard_results, images_dir)

np.save(f"{images_dir}/splitguard_results.npy", splitguard_results)


def plot_log(ax, x, y, label):
    ax.plot(x, y, color="black")
    ax.set(title=label)
    ax.grid()


n = 5
fix, ax = plt.subplots(1, n, figsize=(n * 5, 3))
x = np.arange(0, len(LOG)) * log_frequency

plot_log(ax[0], x, LOG[:, 0], label="Loss $f$")
plot_log(ax[1], x, LOG[:, 1], label="Loss $\\tilde{f}$ and $\\tilde{f}^{-1}$")
plot_log(ax[2], x, LOG[:, 2], label="Loss $D$")
plot_log(ax[3], x, LOG[:, 3], label="Reconstruction error (VALIDATION)")
plot_log(ax[4], x, LOG[:, 4], label="Target loss")

fix.savefig(f"{images_dir}/training_logs.png")


X = torch.from_numpy(
    getImagesDS(reconstruct_eval_dataset, len(reconstruct_eval_dataset))
).cuda()
X_recovered, control = fsha.attack(X)

mse_loss = cal_recon_loss(X, X_recovered)

logging.info(f"===== Reconstruction Loss is {mse_loss:.6f}")


X_cpu = X.cpu().detach().numpy()
X_recovered_cpu = X_recovered.cpu().detach().numpy()

individual_mse = []
for i in range(len(X_cpu)):
    mse = np.mean((X_cpu[i] - X_recovered_cpu[i]) ** 2)
    individual_mse.append(mse)


individual_mse = np.array(individual_mse)
best_indices = np.argsort(individual_mse)[:20]
best_mse_values = individual_mse[best_indices]

X_cpu_best = X_cpu[best_indices]
X_recovered_cpu_best = X_recovered_cpu[best_indices]

np.save(f"{images_dir}/X_cpu_all.npy", X_cpu)
np.save(f"{images_dir}/X_recovered_cpu_all.npy", X_recovered_cpu)
np.save(f"{images_dir}/X_cpu_best20.npy", X_cpu_best)
np.save(f"{images_dir}/X_recovered_cpu_best20.npy", X_recovered_cpu_best)
np.save(f"{images_dir}/best_indices.npy", best_indices)
np.save(f"{images_dir}/best_mse_values.npy", best_mse_values)

logging.info(f"Best 20 images MSE values: {best_mse_values}")
logging.info(f"Best 20 images indices: {best_indices}")
logging.info(f"Average MSE of best 20: {np.mean(best_mse_values):.6f}")

fig = plotX(X_cpu_best, f"{images_dir}/original_samples_best20.png")
fig = plotX(X_recovered_cpu_best, f"{images_dir}/recovered_samples_best20.png")

plot_attack_results(
    X_cpu_best,
    X_recovered_cpu_best,
    os.path.join(images_dir, "attack_results_best20.png"),
)

print("Generating visualizations for all 500 images...")
logging.info("Starting to generate visualizations for all attack results...")

all_results_dir = plot_all_attack_results(
    X_cpu, X_recovered_cpu, images_dir, images_per_figure=50
)
print(f"All visualization files saved in: {all_results_dir}")
