import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch


def set_logging(output_dir, log_file_name):
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    logging.getLogger().handlers.clear()

    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    os.makedirs(output_dir, exist_ok=True)

    log_file = os.path.join(output_dir, log_file_name)

    if os.path.exists(log_file):
        os.remove(log_file)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S"))

    logging.getLogger().addHandler(fh)

    print(f"Log file created: {log_file}")


def get_test_score(m1, m2, dataloader):
    m1.eval()
    m2.eval()
    score = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.cuda()
            labels = labels.long().cuda()

            pred = m2(m1(images))
            predicted = torch.argmax(pred, dim=1)
            total += labels.size(0)
            score += (predicted == labels).sum().item()
    m1.train()
    m2.train()
    return 100 * score / total


def plotX(X, filename):
    n = len(X)
    fig, ax = plt.subplots(1, n, figsize=(n * 3, 3))
    plt.axis("off")
    plt.subplots_adjust(wspace=0, hspace=-0.05)

    for i in range(n):
        img = X[i]
        img = img.squeeze()

        if len(img.shape) == 3:
            img = np.transpose(img, (1, 2, 0))

        img = (img + 1) / 2
        img = np.clip(img, 0, 1)

        ax[i].imshow(img, cmap="inferno")
        ax[i].set(xticks=[], yticks=[])
        ax[i].set_aspect("equal")

    fig.savefig(filename)
    plt.close(fig)
    return fig


def plot_attack_results(X, X_recon, file_name):
    X = np.transpose(X, (0, 2, 3, 1))
    X_recon = np.transpose(X_recon, (0, 2, 3, 1))

    X_normalized = np.clip((X + 1) / 2, 0, 1)
    X_recon_normalized = np.clip((X_recon + 1) / 2, 0, 1)

    n = len(X)
    print("Number of images to plot: ", n)
    fig, ax = plt.subplots(2, n, figsize=(n * 1.3, 3))
    plt.axis("off")
    plt.subplots_adjust(wspace=0.05, hspace=0.1)

    for i in range(n):
        ax[0, i].imshow(X_normalized[i])
        ax[1, i].imshow(X_recon_normalized[i])
        ax[0, i].set(xticks=[], yticks=[])
        ax[1, i].set(xticks=[], yticks=[])

    plt.savefig(file_name, dpi=150, bbox_inches="tight")
    print("The image save path: ", file_name)
    plt.close(fig)
    return fig


def cal_recon_loss(X, X_recovered):
    test_samples = 500
    assert X.shape == X_recovered.shape, "Input tensors must have the same shape"
    assert X.shape[0] == test_samples, f"Input tensors must have {test_samples}  "
    mse_criterion = torch.nn.MSELoss()
    return mse_criterion(X, X_recovered)


def plot_splitguard_result(results, images_dir):
    plt.plot(results, label="attack")
    plt.ylim(0, 1.1)
    plt.xlabel("No. of fake batches")
    plt.ylabel("SG score")
    plt.legend()
    plt.savefig(f"{images_dir}/splitguard.png")


def plot_all_attack_results(X_cpu, X_recovered_cpu, images_dir, images_per_figure=50):

    total_images = len(X_cpu)
    num_figures = (total_images + images_per_figure - 1) // images_per_figure

    all_results_dir = os.path.join(images_dir, "all_attack_results")
    os.makedirs(all_results_dir, exist_ok=True)

    for fig_idx in range(num_figures):
        start_idx = fig_idx * images_per_figure
        end_idx = min(start_idx + images_per_figure, total_images)
        current_batch_size = end_idx - start_idx

        X_batch = X_cpu[start_idx:end_idx]
        X_recon_batch = X_recovered_cpu[start_idx:end_idx]

        X_batch = np.transpose(X_batch, (0, 2, 3, 1))
        X_recon_batch = np.transpose(X_recon_batch, (0, 2, 3, 1))

        X_batch_norm = np.clip((X_batch + 1) / 2, 0, 1)
        X_recon_batch_norm = np.clip((X_recon_batch + 1) / 2, 0, 1)

        fig, ax = plt.subplots(
            2, current_batch_size, figsize=(current_batch_size * 1.5, 3)
        )

        if current_batch_size == 1:
            ax = ax.reshape(2, 1)

        for i in range(current_batch_size):

            ax[0, i].imshow(X_batch_norm[i])
            ax[0, i].set(xticks=[], yticks=[])
            ax[0, i].set_title(f"Orig {start_idx + i}", fontsize=8)

            ax[1, i].imshow(X_recon_batch_norm[i])
            ax[1, i].set(xticks=[], yticks=[])
            ax[1, i].set_title(f"Recon {start_idx + i}", fontsize=8)

        plt.tight_layout()

        save_path = os.path.join(
            all_results_dir,
            f"attack_results_{fig_idx:03d}_{start_idx:03d}-{end_idx-1:03d}.png",
        )
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved batch {fig_idx+1}/{num_figures}: {save_path}")

    logging.info(f"All {total_images} attack results saved to {all_results_dir}")
    return all_results_dir
