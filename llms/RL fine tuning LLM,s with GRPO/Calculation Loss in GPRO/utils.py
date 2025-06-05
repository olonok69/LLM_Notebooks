import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import List
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict  # noqa: F401

import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from openai import OpenAI
from transformers import AutoTokenizer
from tabulate import tabulate
from predibase import Predibase, DeploymentConfig


load_dotenv("../.env")


def generate_output_logps(num_generations, sequence_length, vocab_size):
    # Initialize a random tensor of shape (num_generations, sequence_length, vocab_size)
    # This simulates doing a forward pass with the model
    token_probs = torch.randn(num_generations, sequence_length, vocab_size)

    # We use log_softmax so that for each token position we have a proper log-probability distribution 
    # over the vocabulary. These values represent the model’s confidence in each token.
    token_logps = F.log_softmax(token_probs, dim=-1)

    return token_logps


def plot_token_probability_shift(new_logps, old_logps, gen_idx=0, token_pos=64, top_k=10):
    """
    Plots a comparison of token probabilities at a specific token position in a specific generation,
    comparing new vs. old log probabilities.

    Args:
        new_logps (Tensor): Tensor of new log probabilities with shape (batch, seq_len, vocab_size).
        old_logps (Tensor): Tensor of old log probabilities with shape (batch, seq_len, vocab_size).
        gen_idx (int): Index of the generation in the batch to visualize.
        token_pos (int): Position of the token in the sequence to visualize.
        top_k (int): Number of top tokens (by new probability) to display.
    """
    # Convert log probabilities to probabilities
    new_probs = new_logps[gen_idx, token_pos].exp().detach().numpy()
    old_probs = old_logps[gen_idx, token_pos].exp().detach().numpy()

    # Select top_k tokens by new probability
    top_tokens = np.argsort(new_probs)[-top_k:][::-1]
    top_new_probs = new_probs[top_tokens]
    top_old_probs = old_probs[top_tokens]

    # Compute ratios
    ratios = top_new_probs / (top_old_probs + 1e-10)  # add epsilon to avoid division by zero

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(top_k)
    bar_width = 0.35

    ax.bar(x - bar_width/2, top_old_probs, bar_width, label='Ref Prob')
    ax.bar(x + bar_width/2, top_new_probs, bar_width, label='New Prob')

    for i, ratio in enumerate(ratios):
        ax.text(x[i], max(top_old_probs[i], top_new_probs[i]) + 0.0001,
                f'{ratio:.2f}', ha='center', va='bottom')

    ax.set_xticks(x)
    ax.set_xticklabels([f'Token {tok}' for tok in top_tokens])
    ax.set_ylabel("Probability")
    ax.set_title(f"Token Probabilities at Position {token_pos} (Gen {gen_idx})\nAnnotated with Ratio new/old")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_avg_logprobs_per_position(
    new_per_token_logps: torch.Tensor,
    old_per_token_logps: torch.Tensor,
    title: str = "Avg Log Probability per Token Position"
):
    """
    Plots the average log probabilities per token position for two sets of log probabilities.

    Args:
        new_per_token_logps (torch.Tensor): Tensor of shape (num_generations, seq_len) for new log probs.
        old_per_token_logps (torch.Tensor): Tensor of shape (num_generations, seq_len) for ref log probs.
        title (str): Title for the plot.
    """
    avg_new_logps = new_per_token_logps.mean(dim=0)
    avg_old_logps = old_per_token_logps.mean(dim=0)

    plt.figure(figsize=(10, 5))
    plt.plot(avg_old_logps.numpy(), label="Ref", alpha=0.7)
    plt.plot(avg_new_logps.numpy(), label="New", alpha=0.7)
    plt.xlabel("Token Position")
    plt.ylabel("Avg Log Probability Per Token Position For Generated Tokens")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def visualize_clipped_ratios(ratio_unclipped, ratio_clipped, epsilon):
    # Create boolean mask
    was_clipped = (ratio_unclipped != ratio_clipped).squeeze()

    # Convert to tick/cross symbols
    was_clipped_symbols = ["✓" if clipped else "✗" for clipped in was_clipped.tolist()]

    # Create DataFrame
    df = pd.DataFrame({
        "token_position": list(range(ratio_unclipped.shape[-1])),
        "ratio_unclipped": ratio_unclipped.squeeze().tolist(),
        "ratio_clipped": ratio_clipped.squeeze().tolist(),
        "was_clipped": was_clipped_symbols
    })

    print(df)


def visualize_ratio_clipping(ratio: torch.Tensor, ratio_clipped: torch.Tensor, epsilon: float, zoom_xlim=(0.5, 1.5)):
    """
    Visualize raw vs. clipped probability ratios and highlight clipping behavior.

    Args:
        ratio (torch.Tensor): Raw ratio tensor (exp(new_logp - old_logp)).
        ratio_clipped (torch.Tensor): Clipped ratio tensor.
        epsilon (float): Clipping threshold (e.g., 0.2).
        zoom_xlim (tuple): x-axis limits for zoomed-in histogram.
    """
    ratio_np = ratio.flatten().cpu().numpy()
    ratio_clipped_np = ratio_clipped.flatten().cpu().numpy()

    # Compute clipping stats
    num_total = ratio.numel()
    clipped_mask = (ratio < 1 - epsilon) | (ratio > 1 + epsilon)
    num_clipped = clipped_mask.sum().item()
    percent_clipped = 100 * num_clipped / num_total

    num_unclipped = num_total - num_clipped

    print("📊 Clipping Stats:")
    print(f"   Total tokens:      {num_total}")
    print(f"   Clipped tokens:    {num_clipped} ({percent_clipped:.2f}%)")
    print(f"   Unclipped tokens:  {num_unclipped} ({100 - percent_clipped:.2f}%)")

    # # --- Plot 1: Histogram ---
    # plt.figure(figsize=(12, 5))
    # plt.hist(ratio_np, bins=100, alpha=0.6, label="Raw Ratio", color="steelblue")
    # plt.hist(ratio_clipped_np, bins=100, alpha=0.6, label="Clipped Ratio", color="darkorange")
    # plt.axvline(1 - epsilon, color="red", linestyle="--", label="Clip Min")
    # plt.axvline(1 + epsilon, color="red", linestyle="--", label="Clip Max")
    # plt.xlabel("Probability Ratio (new / ref)")
    # plt.ylabel("Token Count")
    # plt.title("Distribution of Token Probability Ratios (Zoomed In)")
    # plt.xlim(*zoom_xlim)
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    # --- Plot 2: Clipping counts ---
    plt.figure(figsize=(5, 5))
    plt.bar(["Clipped", "Unclipped"], [num_clipped, num_unclipped], color=["red", "green"])
    plt.title("Number of Tokens Clipped vs. Unclipped")
    plt.ylabel("Token Count")
    plt.tight_layout()
    plt.show()


def visualize_per_token_kl(per_token_kl: torch.Tensor):
    """
    Visualizes per-token KL divergence across sequences and token positions.

    Args:
        per_token_kl (torch.Tensor): Tensor of shape (num_generations, sequence_length)
                                     containing per-token KL divergence values.
    """
    avg_kl = per_token_kl.mean(dim=0).detach().cpu().numpy()

    plt.figure(figsize=(10, 4))
    plt.plot(avg_kl, label="Avg KL per Token Pos", color="blue")
    plt.xlabel("Token Position")
    plt.ylabel("KL Divergence")
    plt.title("Average KL Divergence Across Token Positions")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
