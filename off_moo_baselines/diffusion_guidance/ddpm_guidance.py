"""
Diffusion Guidance Training and Sampling

This module contains the training functions and diffusion process implementation
for multi-objective optimization using diffusion models. It includes functions
for training unconditional diffusion models, preference models, and guided
sampling with preference guidance.
"""

import os
import copy
import math
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from utils import *
from off_moo_baselines.diffusion_guidance.modules import (
    EMA,
    Model_unconditional,
    Preference_model,
    save_model,
    load_model,
)
import logging
from torch.func import functional_call, vmap, grad


# =============================================================================
# Logging Configuration
# =============================================================================


logging.basicConfig(
    format="%(asctime)s - %(levelname)s: %(message)s",
    level=logging.INFO,
    datefmt="%I:%M:%S",
)


# =============================================================================
# Training Functions
# =============================================================================


def train(dataloader, model=None, diffusion=None, ema=None):
    """
    Train an unconditional diffusion model.

    This function trains a diffusion model to predict noise given noisy inputs
    and timesteps. It uses MSE loss and includes EMA updates for model stability.

    Args:
        dataloader: DataLoader providing (images, labels, hvs) batches
        model: Pre-trained model (if None, creates new Model_unconditional)
        diffusion: Diffusion process (if None, creates new Diffusion)
        ema: EMA object (if None, creates new EMA with beta=0.99)

    Returns:
        tuple: (trained_model, diffusion)
    """
    X_size = dataloader.dataset[0][0].shape[-1]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if model is None:
        model = Model_unconditional(dim=X_size).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=5e-4)
    if diffusion is None:
        diffusion = Diffusion(img_size=X_size, device=device)
    l = len(dataloader)
    if ema is None:
        ema = EMA(0.99)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    for epoch in range(200):
        loss_epoch = []
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (data, labels, hvs) in enumerate(pbar):
            data = data.to(device)
            labels = labels.to(device)
            data = data * 2 - 1
            hvs = hvs.to(device)
            t = diffusion.sample_timesteps(data.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(data, t)
            predicted_noise = model(x_t, t)
            loss = ((noise - predicted_noise).pow(2).sum(-1)).mean()
            loss_epoch.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            ema.step_ema(ema_model, model)

            pbar.set_postfix(MSE=loss.item())
        logging.info(f"Epoch {epoch} loss: {np.mean(loss_epoch)}")
    return model, diffusion


def train_preference(
    dataloader,
    model=None,
    diffusion=None,
    val_loader=None,
    config=None,
    tolerance=50,
    model_save_path=None,
    three_dim_out=False,
):
    """
    Train a preference model for guided diffusion sampling.

    This function trains a model to predict preferences between pairs of solutions.
    The model learns to rank solutions based on their quality, which is then used
    for guided sampling in the diffusion process.

    Args:
        dataloader: DataLoader providing (x_1, x_2, y) batches where y indicates preference
        model: Pre-trained preference model (if None, creates new Preference_model)
        diffusion: Diffusion process (if None, creates new Diffusion)
        val_loader: Validation dataloader for early stopping
        config: Configuration dictionary
        tolerance (int): Number of epochs without improvement before early stopping
        model_save_path (str): Path to save the best model
        three_dim_out (bool): Whether model outputs 3D tensor

    Returns:
        tuple: (trained_model, diffusion)
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    if model is None:
        model = Preference_model(
            input_dim=dataloader.dataset[0][0].shape[-1],
            device=device,
            three_dim_out=three_dim_out,
        ).to(device)
    if diffusion is None:
        diffusion = Diffusion(
            img_size=dataloader.dataset[0][0].shape[-1], device=device
        ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    best_val_loss = 1e10
    curr_tol = 0
    for epoch in range(2000):
        loss_epoch = []
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (x_1, x_2, y) in enumerate(pbar):
            x_1 = x_1.to(device)
            x_2 = x_2.to(device)
            y = y.to(device)
            x_1 = x_1 * 2 - 1
            x_2 = x_2 * 2 - 1
            t = diffusion.sample_timesteps(x_1.shape[0]).to(device)
            x_1, _ = diffusion.noise_images(x_1, t)
            x_2, _ = diffusion.noise_images(x_2, t)
            pred = model(x_1, x_2, t)
            loss_1 = loss_fn(pred, y.squeeze().long())
            loss = torch.mean(loss_1)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(MSE=loss.item())
            loss_epoch.append(loss.item())

        if val_loader is not None:
            model.eval()
            if epoch % 5 == 0:
                val_loss = []
                for epoch in range(5):
                    for i, (x_1, x_2, y) in enumerate(val_loader):
                        x_1 = x_1.to(device)
                        x_2 = x_2.to(device)
                        x_1 = x_1 * 2 - 1
                        x_2 = x_2 * 2 - 1
                        t = diffusion.sample_timesteps(x_1.shape[0]).to(device)
                        x_1, _ = diffusion.noise_images(x_1, t)
                        x_2, _ = diffusion.noise_images(x_2, t)
                        y = y.to(device)
                        pred = model(x_1, x_2, t)
                        loss_1 = loss_fn(pred, y.squeeze().long())
                        loss = torch.mean(loss_1)
                        val_loss.append(loss.item())
                model.train()
                logging.info(
                    f"Epoch {epoch} loss: {np.mean(loss_epoch)} val_loss: {np.mean(val_loss)}"
                )
                if np.mean(val_loss) < best_val_loss:
                    best_val_loss = np.mean(val_loss)
                    save_model(model, model_save_path, device)
                    curr_tol = 0
                else:
                    curr_tol += 1
                    if curr_tol > tolerance:
                        break
        logging.info(f"Epoch {epoch} loss: {np.mean(loss_epoch)}")
    return model


# =============================================================================
# Diffusion Process
# =============================================================================


class Diffusion:
    """
    Diffusion process for gradual noise addition and removal.

    This class implements the forward and reverse diffusion processes used in
    diffusion models. It handles noise scheduling, forward diffusion (adding noise),
    and reverse diffusion (removing noise) with optional preference guidance.

    Attributes:
        noise_steps (int): Number of diffusion steps
        beta_start (float): Initial noise level
        beta_end (float): Final noise level
        beta (torch.Tensor): Noise schedule
        alpha (torch.Tensor): 1 - beta
        alpha_hat (torch.Tensor): Cumulative product of alpha
        img_size (int): Size of input images/vectors
        device (str): Device to run on
    """

    def __init__(
        self,
        noise_steps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        img_size=256,
        device="cuda",
    ):
        """
        Initialize the diffusion process.

        Args:
            noise_steps (int): Number of diffusion steps (default: 1000)
            beta_start (float): Initial noise level (default: 1e-4)
            beta_end (float): Final noise level (default: 0.02)
            img_size (int): Size of input images/vectors (default: 256)
            device (str): Device to run on (default: "cuda")
        """
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.device = device

    def prepare_noise_schedule(self):
        """
        Prepare the noise schedule for diffusion.

        Returns:
            torch.Tensor: Linear noise schedule from beta_start to beta_end
        """
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        """
        Add noise to images according to the diffusion schedule.

        Args:
            x (torch.Tensor): Input images of shape (batch_size, img_size)
            t (torch.Tensor): Timesteps of shape (batch_size,)

        Returns:
            tuple: (noisy_images, noise) where both have shape (batch_size, img_size)
        """
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        """
        Sample random timesteps for training.

        Args:
            n (int): Number of timesteps to sample

        Returns:
            torch.Tensor: Random timesteps of shape (n,)
        """
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample_with_preference(
        self,
        model,
        n,
        preference_model,
        best_x_data,
        cfg_scale=3,
        return_latents=False,
        ddim=False,
    ):
        """
        Sample new images using preference-guided diffusion.

        This method performs reverse diffusion sampling with preference guidance.
        It uses a preference model to guide the sampling process towards solutions
        that are close to the pareto front.

        Args:
            model: Trained diffusion model for noise prediction
            n (int): Number of samples to generate
            preference_model: Trained preference model for guidance
            best_x_data (torch.Tensor): Best known solutions for comparison
            cfg_scale (float): Scale factor for preference guidance (default: 3)
            return_latents (bool): Whether to return intermediate latents (default: False)
            ddim (bool): Whether to use DDIM sampling (default: False)

        Returns:
            torch.Tensor: Generated samples of shape (n, img_size)
        """

        def compute_grad(x, best_x_data, t, preference_model, params, buffers):
            """
            Compute preference gradient for guided sampling.

            Args:
                x: Current sample
                best_x_data: Best known solutions in the training set
                t: Current timestep
                preference_model: Preference model
                params: Model parameters
                buffers: Model buffers

            Returns:
                torch.Tensor: Preference gradient
            """
            x = x.unsqueeze(0)
            best_x_data = best_x_data.unsqueeze(0)
            predictions = functional_call(
                preference_model, (params, buffers), (x, best_x_data, t)
            )
            pref_logits = torch.nn.functional.log_softmax(predictions, dim=-1)
            pref_logits = pref_logits[..., 0].squeeze()
            return pref_logits

        params = {k: v.detach() for k, v in preference_model.named_parameters()}
        buffers = {k: v.detach() for k, v in preference_model.named_buffers()}
        logging.info(f"Sampling {n} new images....")
        model.eval()
        preference_model.eval()
        latents = []

        x = torch.randn((n, self.img_size)).to(self.device)
        best_x_data = best_x_data.to(x.dtype).to(self.device)
        best_x_data = 2 * best_x_data - 1
        for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
            if i % 10 == 0 and return_latents:
                latents.append(x)
            t = (torch.ones(n) * i).long().to(self.device)
            with torch.no_grad():
                predicted_noise = model(x, t)
            if cfg_scale > 0:
                x_ = x.clone()
                x_ = x_.detach().requires_grad_(True)
                score = vmap(grad(compute_grad), (0, 0, 0, None, None, None))(
                    x_, best_x_data, t, preference_model, params, buffers
                )
                best_x_data = x_.detach()
                score = score.detach()
            else:
                score = 0
            alpha = self.alpha[t][:, None]
            alpha_hat = self.alpha_hat[t][:, None]
            alpha_hat_t_1 = self.alpha_hat[t - 1][:, None]
            beta = self.beta[t][:, None]
            if i > 1 and not ddim:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            if ddim:
                x = (
                    (torch.sqrt(alpha_hat_t_1) / torch.sqrt(alpha_hat))
                    * (x - (torch.sqrt(1 - alpha_hat) * predicted_noise))
                ) + torch.sqrt(1 - alpha_hat_t_1) * cfg_scale * score
            else:
                x = (
                    1
                    / torch.sqrt(alpha)
                    * (
                        x
                        - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise
                    )
                    + torch.sqrt(beta) * noise
                    + cfg_scale * score * beta
                )
            if i % 100 == 0:
                x = x.clamp(-1, 1)
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        latents = [(latent.clamp(-1, 1) + 1) / 2 for latent in latents]
        latents.append(x)
        if return_latents:
            return x, torch.stack(latents)
        # x = (x * 255).type(torch.uint8)
        return x
