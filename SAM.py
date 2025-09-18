import itertools
import logging
import math
import pickle
import random
import statistics

import numpy as np
import torch
import torch.nn as nn
import tqdm

import architectures_torch as architectures
from architectures_torch import split_resnet_client, split_resnet_server
from utils import *

fake, r_1, r_2, results, fake_indices = [], [], [], [], []
# index_global=0
mult = 5
exp = 2
N = 20
p_fake = 0.1
b_fake = 64
latency = 1000


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
        m.bias.data.zero_()
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
        m.bias.data.zero_()


def distance_data_loss(a, b):
    l = nn.MSELoss()
    return l(a, b)


def distance_data(a, b):
    l = nn.MSELoss()
    return l(a, b)


def zeroing_grad(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            param.grad = torch.zeros_like(param.grad).to(param.device)


def angle(v1, v2):
    unit_vector_1 = v1 / np.linalg.norm(v1)
    unit_vector_2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    return np.arccos(dot_product)


def sigmoid(x, shift=0, mult=1, exp=1):
    x_p = (x - shift) * mult
    return (1 / (1 + np.exp(-x_p))) ** exp


def sg_score(fakes, controls, regulars, shift=0, mult=1, exp=1, raw=False):
    f_mean = sum(fakes) / len(fakes)
    c_mean = sum(controls) / len(controls)
    r_mean = sum(regulars) / len(regulars)
    cr_mean = (c_mean + r_mean) / 2

    f_mean_mag = sum([np.linalg.norm(v) for v in fakes]) / len(fakes)
    c_mean_mag = sum([np.linalg.norm(v) for v in controls]) / len(controls)
    r_mean_mag = sum([np.linalg.norm(v) for v in regulars]) / len(regulars)
    cr_mean_mag = (c_mean_mag + r_mean_mag) / 2

    mag_div = abs(f_mean_mag - cr_mean_mag) + abs(c_mean_mag - r_mean_mag)

    x = angle(f_mean, cr_mean) * (abs(f_mean_mag - cr_mean_mag) / mag_div) - angle(
        c_mean, r_mean
    ) * (abs(r_mean_mag - c_mean_mag) / mag_div)

    if raw:
        return x
    else:
        return sigmoid(x, shift=shift, mult=mult, exp=exp)


class SAM:

    def loadBiasNetwork(self, make_decoder, z_shape, channels):
        return make_decoder(z_shape, channels=channels)

    def __init__(
        self,
        xpriv,
        xpub,
        id_setup,
        batch_size,
        hparams,
        num_classes,
        reconstruct_eval_dataset=None,
    ):
        input_shape = xpriv[0][0].shape

        self.hparams = hparams
        self.num_classes = num_classes
        self.reconstruct_eval_dataset = reconstruct_eval_dataset

        # setup dataset
        self.client_dataset = torch.utils.data.DataLoader(
            xpriv, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        self.attacker_dataset = torch.utils.data.DataLoader(
            xpub, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
        )

        self.batch_size = batch_size
        make_f, make_tilde_f, make_decoder, make_D = architectures.SETUPS[id_setup]

        self.f = make_f(input_shape)
        self.tilde_f = make_tilde_f(input_shape)

        test_input = torch.zeros([1, input_shape[0], input_shape[1], input_shape[2]])
        f_out = self.f(test_input)
        tilde_f_out = self.tilde_f(test_input)

        print(f_out.size()[1:], tilde_f_out.size()[1:])

        assert f_out.size()[1:] == tilde_f_out.size()[1:]
        z_shape = tilde_f_out.size()[1:]
        print(z_shape)
        self.D = make_D(z_shape)

        self.target_net = split_resnet_server(
            input_shape=z_shape, split_level=id_setup + 1, num_classes=num_classes
        )

        self.decoder = self.loadBiasNetwork(
            make_decoder, z_shape, channels=input_shape[0]
        )

        # initialize modules
        self.f.apply(init_weights)
        self.tilde_f.apply(init_weights)
        self.D.apply(init_weights)
        self.decoder.apply(init_weights)
        self.target_net.apply(init_weights)
        # move models to GPU
        self.f.cuda()
        self.tilde_f.cuda()
        self.D.cuda()
        self.decoder.cuda()
        self.target_net.cuda()
        # setup optimizers
        self.optimizer0 = torch.optim.Adam(self.f.parameters(), lr=hparams["lr_f"])
        self.optimizer1 = torch.optim.Adam(
            [
                {"params": self.tilde_f.parameters()},
                {"params": self.decoder.parameters()},
            ],
            lr=hparams["lr_tilde"],
        )
        self.optimizer2 = torch.optim.Adam(self.D.parameters(), lr=hparams["lr_D"])
        self.optimizer3 = torch.optim.Adam(
            self.target_net.parameters(), lr=hparams["lr_f"]
        )
        self.index_global = 0

    @staticmethod
    def addNoise(x, alpha):
        if alpha == 0.0:
            return x

        noise = torch.randn(x.size(), device=x.device, dtype=x.dtype) * alpha
        return x + noise

    def _eval_classification(self):
        self.f.eval()
        self.target_net.eval()

        total_correct = 0
        total_samples = 0
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for batch_idx, (x_public, label_public) in enumerate(self.attacker_dataset):
                x_public = x_public.cuda()
                label_public = label_public.long().cuda()

                if label_public.max() >= self.num_classes:
                    raise ValueError("Labels exceed number of classes")

                z_public = self.f(x_public)

                outputs = self.target_net(z_public)

                loss = criterion(outputs, label_public)
                total_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total_samples += label_public.size(0)
                total_correct += (predicted == label_public).sum().item()

        accuracy = 100.0 * total_correct / total_samples if total_samples > 0 else 0.0
        avg_loss = total_loss / max(1, len(self.attacker_dataset))

        self.f.train()
        self.target_net.train()

        return accuracy, avg_loss

    def train_step(self, x_private, x_public, label_private, label_public):
        # Set global variables (consider if these are truly necessary to be global)
        global N, p_fake, fake, r_1, r_2, results, fake_indices, mult, exp, b_fake, latency

        # Initialize variables
        train_acc = 0.0
        train_loss = 0.0
        f_losses, tilde_f_losses, D_losses, losses_c_verification, target_losses = (
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )

        # Set models to training mode
        self.f.train()
        self.tilde_f.train()
        self.decoder.train()
        self.D.train()
        self.target_net.train()

        # Move data to GPU
        x_private = x_private.cuda(non_blocking=False)
        x_public = x_public.cuda(non_blocking=False)

        send_fakes = self.index_global > N and random.random() <= p_fake

        # Prepare labels
        LABELS_SENT = label_private
        if send_fakes:
            offset = random.randint(1, self.num_classes - 1)
            LABELS_SENT = (label_private + offset) % self.num_classes

        LABELS_SENT = LABELS_SENT.long().cuda()
        label_public = label_public.long().cuda()

        target_criterion = nn.CrossEntropyLoss()

        ##### Train f and target_net #####

        z_private_for_task = self.f(x_private)
        z_private_for_adv = z_private_for_task

        outputs = self.target_net(z_private_for_task)
        target_loss = target_criterion(outputs, LABELS_SENT)

        _, train_pred = torch.max(outputs, 1)
        train_acc += (train_pred.detach() == LABELS_SENT.detach()).sum().item()
        train_loss += target_loss.item()
        print(
            "Train Acc: {:3.6f} Loss: {:3.6f}".format(
                train_acc / len(x_private), train_loss / len(x_private)
            )
        )

        self.optimizer0.zero_grad()
        self.optimizer3.zero_grad()

        if self.index_global <= latency:
            target_loss.backward()
        else:
            # Define adversarial losses (SERVER-SIDE)
            adv_private_logits = self.D(z_private_for_adv)

            if self.hparams["WGAN"]:
                f_loss = (
                    torch.mean(adv_private_logits) + self.hparams["alpha"] * target_loss
                )
            else:
                sigmoid = nn.Sigmoid()
                criterion = nn.BCELoss()
                f_loss = torch.mean(
                    torch.ones_like(adv_private_logits.detach()) - adv_private_logits
                )

            # Backward passes
            f_loss.backward()
            zeroing_grad(self.D)

        # Gradient handling and update
        f_first_param = list(self.f.parameters())[0]

        if f_first_param.grad is not None:
            client_grad = f_first_param.grad.detach().clone().flatten()
        else:
            client_grad = torch.zeros_like(f_first_param.view(-1))
            print(
                f"Warning: f.grad is None at iteration {self.index_global}, using zero gradient for SplitGuard"
            )

        if send_fakes:
            fake.append(client_grad.cpu())
            fake_indices.append(self.index_global)
            if len(r_1) > 0 and len(r_2) > 0:
                sg = sg_score(fake, r_1, r_2, mult=mult, exp=exp, raw=False)
                results.append(sg)
        elif self.index_global > N:
            if random.random() <= 0.5:
                r_1.append(client_grad.cpu())
            else:
                r_2.append(client_grad.cpu())

        if not send_fakes:
            self.optimizer0.step()
        else:
            self.optimizer0.zero_grad()
        self.optimizer3.step()

        ##### Train ~f、decoder #####

        # Public data processing for invertibility loss
        z_public = self.tilde_f(x_public)
        rec_x_public = self.decoder(z_public)
        public_rec_loss = distance_data_loss(x_public, rec_x_public)

        z_private = z_private_for_adv
        # Supervised invertibility loss on private data
        sup_rec_x_priv = self.decoder(z_private.detach())
        sup_z_priv = self.tilde_f(sup_rec_x_priv)
        sup_z_priv_loss = distance_data_loss(sup_z_priv, z_private.detach())

        # Dynamically define the tilde_f_loss based on latency
        if self.index_global <= latency:
            tilde_f_loss = public_rec_loss + 10 * sup_z_priv_loss
        else:
            sup_outputs = self.target_net(sup_z_priv)
            sup_target_loss = target_criterion(sup_outputs, LABELS_SENT)
            tilde_f_loss = public_rec_loss + sup_target_loss

        self.optimizer1.zero_grad()
        tilde_f_loss.backward()
        self.optimizer1.step()
        zeroing_grad(self.target_net)

        ##### Train D #####
        if self.index_global > latency:
            with torch.no_grad():
                z_private_for_D = self.f(x_private)
                z_public_for_D = self.tilde_f(x_public)

            adv_public_logits = self.D(z_public_for_D)
            adv_private_logits_detached = self.D(z_private_for_D)

            if self.hparams["WGAN"]:
                D_loss = torch.mean(adv_public_logits) - torch.mean(
                    adv_private_logits_detached
                )
            else:
                sigmoid = nn.Sigmoid()
                criterion = nn.BCELoss()
                loss_discr_true = -criterion(
                    sigmoid(adv_public_logits),
                    torch.zeros_like(adv_public_logits.detach()),
                )
                loss_discr_fake = -criterion(
                    sigmoid(adv_private_logits_detached),
                    torch.ones_like(adv_private_logits_detached.detach()),
                )
                D_loss = (loss_discr_true + loss_discr_fake) / 2

            if "gradient_penalty" in self.hparams:
                w = float(self.hparams["gradient_penalty"])
                D_gradient_penalty = self.gradient_penalty(
                    z_private.detach(), z_public.detach()
                )
                D_loss += D_gradient_penalty * w

            self.optimizer2.zero_grad()
            D_loss.backward()
            self.optimizer2.step()

        else:
            D_loss = torch.tensor(0.0).cuda()

        # Attack validation
        with torch.no_grad():
            rec_x_private = self.decoder(z_private)
            losses_c_verification = (
                distance_data(x_private, rec_x_private).detach().cpu()
            )

        # Log losses
        target_losses = target_loss.detach().cpu()
        tilde_f_losses = tilde_f_loss.detach().cpu()
        if self.index_global > latency:
            D_losses = D_loss.detach().cpu()
            f_losses = f_loss.detach().cpu()
        else:
            f_losses = target_loss.detach().cpu()

        del target_loss, tilde_f_loss
        if self.index_global > latency:
            del f_loss, D_loss

        self.index_global += 1
        return f_losses, tilde_f_losses, D_losses, losses_c_verification, target_losses

    def gradient_penalty(self, x, x_gen):
        epsilon = torch.rand([x.shape[0], 1, 1, 1]).cuda()
        x_hat = epsilon * x + (1 - epsilon) * x_gen
        x_hat = torch.autograd.Variable(x_hat, requires_grad=True)
        from torch.autograd import grad

        d_hat = self.D(x_hat)
        gradients = grad(
            outputs=d_hat,
            inputs=x_hat,
            grad_outputs=torch.ones_like(d_hat).cuda(),
            retain_graph=True,
            create_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_norm = gradients.norm(2, dim=1)
        penalty = ((gradient_norm - 1) ** 2).mean()
        return penalty

    def attack(self, x_private):
        with torch.no_grad():
            # smashed data sent from the client:
            z_private = self.f(x_private)
            # recover private data from smashed data
            tilde_x_private = self.decoder(z_private)

            z_private_control = self.tilde_f(x_private)
            control = self.decoder(z_private_control)
        return tilde_x_private, control

    def pretrain_autoencoder(self, pretrain_epochs=10, pretrain_lr=0.001, verbose=True):
        """
        Pretrain the autoencoder (tilde_f and decoder) using only the public dataset.

        Args:
            pretrain_epochs: Number of pretraining epochs
            pretrain_lr: Pretraining learning rate
            verbose: Whether to output training logs
        """
        logging.info("Start pretraining autoencoder...")

        # Set to training mode
        self.tilde_f.train()
        self.decoder.train()

        # Create pretraining optimizer (with higher learning rate)
        pretrain_optimizer = torch.optim.Adam(
            list(self.tilde_f.parameters()) + list(self.decoder.parameters()),
            lr=pretrain_lr,
        )

        # Pretraining loss function
        mse_loss = nn.MSELoss()

        # Record pretraining losses
        pretrain_losses = []

        for epoch in range(pretrain_epochs):
            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, (x_public, _) in enumerate(self.attacker_dataset):
                x_public = x_public.cuda()

                z_public = self.tilde_f(x_public)
                rec_x_public = self.decoder(z_public)

                basic_recon_loss = mse_loss(rec_x_public, x_public)

                total_loss = basic_recon_loss

                pretrain_optimizer.zero_grad()
                total_loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    list(self.tilde_f.parameters()) + list(self.decoder.parameters()),
                    max_norm=1.0,
                )

                pretrain_optimizer.step()

                epoch_loss += total_loss.item()
                num_batches += 1

            avg_loss = epoch_loss / max(1, num_batches)
            pretrain_losses.append(avg_loss)

            logging.info(
                f"Pretraining Epoch [{epoch+1}/{pretrain_epochs}], Reconstruction Loss: {avg_loss:.6f}"
            )

        logging.info(
            f"Pretraining Completed! Final Reconstruction Loss: {pretrain_losses[-1]:.6f}"
        )
        return pretrain_losses

    def __call__(self, iterations, log_frequency=500, verbose=False, progress_bar=True):

        n = int(iterations / log_frequency)
        LOG = np.zeros((n, 5))

        client_iterator = iter(self.client_dataset)
        attacker_iterator = iter(self.attacker_dataset)
        print("RUNNING...")
        iterator = list(range(iterations))
        j = 0

        self.pretrain_autoencoder()

        for i in tqdm.tqdm(iterator, total=iterations):
            try:
                x_private, label_private = next(client_iterator)
                if x_private.size(0) != self.batch_size:
                    client_iterator = iter(self.client_dataset)
                    x_private, label_private = next(client_iterator)
            except StopIteration:
                client_iterator = iter(self.client_dataset)
                x_private, label_private = next(client_iterator)
            try:
                x_public, label_public = next(attacker_iterator)
                if x_public.size(0) != self.batch_size:
                    attacker_iterator = iter(self.attacker_dataset)
                    x_public, label_public = next(attacker_iterator)
            except StopIteration:
                attacker_iterator = iter(self.attacker_dataset)
                x_public, label_public = next(attacker_iterator)
            # log = self.train_step(x_private, x_public, label_private, label_public)

            log = self.train_step(x_private, x_public, label_private, label_public)
            # logging.info("Train_step_alternating!")
            # logging.info(log)

            if i == 0:
                VAL = log[3]
            else:
                VAL += log[3] / log_frequency

            if i % log_frequency == 0:
                torch.save(self.f, "./model/f_model.ckpt")
                torch.save(self.target_net, "./model/target_model.ckpt")
                torch.save(self.tilde_f, "./model/tilde_f_model.ckpt")
                torch.save(self.D, "./model/D_model.ckpt")
                LOG[j] = log

                try:
                    eval_acc, eval_loss = self._eval_classification()
                    logging.info(
                        f"===== Evaluation Accuracy: {eval_acc:.2f}%, Eval Loss: {eval_loss:.6f}"
                    )
                except:
                    logging.info("===== Evaluation Failed =====")

                if verbose:
                    logging.info(
                        "log--%02d%%-%07d] validation: %0.4f"
                        % (int(i / iterations * 100), i, VAL)
                    )
                    logging.info(
                        "f_Loss: {}\nf_tilde_loss: {}\nD_loss: {}\ntarget_loss: {}\n".format(
                            log[0], log[1], log[2], log[4]
                        )
                    )
                    logging.info("losss_c_verification (recon): {}\n".format(log[3]))
                VAL = 0
                j += 1

        return LOG


# ----------------------------------------------------------------------------------------------------------------------
