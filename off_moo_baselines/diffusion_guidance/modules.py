"""
Diffusion Guidance Modules

This module contains the core neural network models and utilities for diffusion-based
multi-objective optimization. It includes models for unconditional generation,
preference learning, and utility functions for model management.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Exponential Moving Average (EMA)
# =============================================================================


class EMA:
    """
    Exponential Moving Average (EMA) for model parameter smoothing.

    EMA helps stabilize training by maintaining a moving average of model parameters,
    which typically leads to better generalization and more stable inference.

    Attributes:
        beta (float): EMA decay rate (0 < beta < 1)
        step (int): Current training step counter
    """

    def __init__(self, beta):
        """
        Initialize EMA with decay rate.

        Args:
            beta (float): EMA decay rate. Higher values (closer to 1) make the
                         average more stable but slower to adapt.
        """
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        """
        Update the moving average model parameters.

        Args:
            ma_model: The moving average model to update
            current_model: The current model with latest parameters
        """
        for current_params, ma_params in zip(
            current_model.parameters(), ma_model.parameters()
        ):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        """
        Compute the exponential moving average of two values.

        Args:
            old: Previous average value
            new: New value to incorporate

        Returns:
            Updated average value
        """
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        """
        Perform one step of EMA update.

        Before step_start_ema, the EMA model is reset to match the current model.
        After step_start_ema, the EMA model is updated using exponential moving average.

        Args:
            ema_model: The moving average model to update
            model: The current model
            step_start_ema (int): Step at which to start EMA updates (default: 2000)
        """
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        """
        Reset EMA model parameters to match current model.

        Args:
            ema_model: The moving average model to reset
            model: The current model to copy parameters from
        """
        ema_model.load_state_dict(model.state_dict())


# =============================================================================
# Diffusion Models
# =============================================================================


class Model_unconditional(nn.Module):
    """
    Unconditional diffusion model for noise prediction.

    This model predicts noise given a noisy input and timestep. It uses a simple
    MLP architecture with sinusoidal positional encoding for timesteps.

    Attributes:
        device (str): Device to run the model on
        dim (int): Input/output dimension
        save_path (str): Path to save model checkpoints
        mlp (nn.Sequential): Main neural network layers
        time_embed (nn.Linear): Time embedding layer
    """

    def __init__(self, dim=256, device="cuda", save_path=None):
        """
        Initialize the unconditional diffusion model.

        Args:
            dim (int): Input/output dimension (default: 256)
            device (str): Device to run the model on (default: "cuda")
            save_path (str): Path to save model checkpoints (default: None)
        """
        super().__init__()
        self.device = device
        self.dim = dim
        self.save_path = save_path
        self.mlp = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, dim),
            nn.LayerNorm(dim),
        )
        self.time_embed = nn.Linear(1, dim)

    def pos_encoding(self, t, dim):
        """
        Generate sinusoidal positional encoding for timesteps.

        Args:
            t (torch.Tensor): Timestep tensor of shape (batch_size, 1)
            dim (int): Dimension of the positional encoding

        Returns:
            torch.Tensor: Positional encoding of shape (batch_size, dim)
        """
        half_dim = dim // 2
        freq = torch.exp(
            math.log(10000)
            * (torch.arange(0, half_dim, device=self.device).float() / half_dim)
        ).to(self.device)
        pos_enc_a = torch.sin(t.repeat(1, half_dim) * freq.unsqueeze(0))
        pos_enc_b = torch.cos(t.repeat(1, half_dim) * freq.unsqueeze(0))
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        if dim % 2:
            pos_enc = torch.cat([pos_enc, torch.zeros_like(pos_enc[:, :1])], dim=-1)
        return pos_enc

    def forward(self, x, t):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Noisy input of shape (batch_size, dim)
            t (torch.Tensor): Timestep tensor of shape (batch_size,)

        Returns:
            torch.Tensor: Predicted noise of shape (batch_size, dim)
        """
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.dim)
        output = self.mlp(x + t)
        return output


class Preference_model(nn.Module):
    """
    Preference model for learning pairwise preferences between designs.

    This model takes two noisy designs and a timestep, and predicts which
    design is preferred. It's used for guided diffusion sampling.

    Attributes:
        device (str): Device to run the model on
        input_dim (int): Input dimension
        save_path (str): Path to save model checkpoints
        three_dim_out (bool): Whether to output 3D tensor
        mlp (nn.Sequential): Main neural network layers
        time_embed (nn.Sequential): Time embedding layers
        preference (nn.Sequential): Preference prediction layers
        out_1 (nn.Linear): First output layer
        out_2 (nn.Linear): Second output layer (if three_dim_out=True)
    """

    def __init__(
        self, input_dim=256, device="cuda", save_path=None, three_dim_out=False
    ):
        """
        Initialize the preference model.

        Args:
            input_dim (int): Input dimension (default: 256)
            device (str): Device to run the model on (default: "cuda")
            save_path (str): Path to save model checkpoints (default: None)
            three_dim_out (bool): Whether to output 3D tensor (default: False)
        """
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.save_path = save_path
        self.three_dim_out = three_dim_out
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.LayerNorm(input_dim),
        )
        self.time_embed = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.LayerNorm(input_dim),
        )
        self.preference = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
        )
        self.out_1 = nn.Linear(512, 1)
        if self.three_dim_out:
            self.out_2 = nn.Linear(512, 1)

    def pos_encoding(self, t, dim):
        """
        Generate sinusoidal positional encoding for timesteps.

        Args:
            t (torch.Tensor): Timestep tensor of shape (batch_size, 1)
            dim (int): Dimension of the positional encoding

        Returns:
            torch.Tensor: Positional encoding of shape (batch_size, dim)
        """
        half_dim = dim // 2
        freq = torch.exp(
            math.log(10000)
            * (torch.arange(0, half_dim, device=self.device).float() / half_dim)
        ).to(self.device)
        pos_enc_a = torch.sin(t.repeat(1, half_dim) * freq.unsqueeze(0))
        pos_enc_b = torch.cos(t.repeat(1, half_dim) * freq.unsqueeze(0))
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        if dim % 2 == 1:
            pos_enc = torch.cat([pos_enc, torch.zeros_like(pos_enc[:, :1])], dim=-1)
        return pos_enc

    def forward(self, x_1, x_2, t):
        """
        Forward pass of the preference model.

        Args:
            x_1 (torch.Tensor): First noisy solution of shape (batch_size, input_dim)
            x_2 (torch.Tensor): Second noisy solution of shape (batch_size, input_dim)
            t (torch.Tensor): Timestep tensor of shape (batch_size,)

        Returns:
            torch.Tensor: Preference prediction of shape (batch_size,) or (batch_size, 2) if three_dim_out=True
        """
        t = t.unsqueeze(-1).type(torch.float)
        em = torch.stack([x_1, x_2], axis=1)
        # em_1 = self.mlp(x_1)
        # em_2 = self.mlp(x_2)
        # em = em_1 + em_2
        t = self.time_embed(self.pos_encoding(t, self.input_dim)).unsqueeze(-2)
        em = self.mlp(em + t)
        # t = t.unsqueeze(-1).type(torch.float)
        output = self.out_1(self.preference(em + t))
        if self.three_dim_out:
            output_2 = self.out_2(self.preference(em + t)).sum(-2, keepdim=True)
            output = torch.cat([output, output_2], dim=-2)
        output = output.squeeze(-1)
        return output


# =============================================================================
# Model Utility Functions
# =============================================================================


def save_model(model, save_path, device="cuda"):
    """
    Save a model's state dict to disk.

    Args:
        model: PyTorch model to save
        save_path (str): Path where to save the model
        device (str): Device to move model back to after saving (default: "cuda")
    """
    torch.save(model.to("cpu").state_dict(), save_path)
    model.to(device)


def load_model(model, save_path, device="cuda"):
    """
    Load a model's state dict from disk.

    Args:
        model: PyTorch model to load state dict into
        save_path (str): Path to the saved model file
        device (str): Device to move model to after loading (default: "cuda")
    """
    model.load_state_dict(torch.load(save_path))
    model.to(device)
    print(f"Successfully load trained model from {save_path}")
