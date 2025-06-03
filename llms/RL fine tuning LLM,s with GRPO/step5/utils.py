import os
import numpy as np
from tabulate import tabulate


MODEL_NAME = "predibase/Meta-Llama-3.1-8B-Instruct-dequantized"

def compute_advantages(rewards: list):
    rewards = np.array(rewards)
    
    # Compute the mean and standard deviation of the rewards
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)

    # Avoid division by zero in case of zero variance (typically happens when all rewards are 0)
    if std_reward == 0:
        return [0] * len(rewards)

    # Divide by stddev of rewards to normalize range to 0
    advantages = (rewards - mean_reward) / std_reward
    return advantages.tolist()


def print_quiz_table(all_answers, rewards):
    advantages = compute_advantages(rewards)
    length = len(all_answers)
    elems = list(zip(range(length), rewards, advantages))

    headers = ["Index", "Reward", "Advantage"]
    table = tabulate(elems, headers=headers, tablefmt="grid").split("\n")
    for row in table:
        print(row)


def print_length_table(lengths, rewards):
    advantages = compute_advantages(rewards)
    length = len(lengths)
    elems = list(zip(range(length), lengths, rewards, advantages))

    headers = ["Index", "Length", "Reward", "Advantage"]
    table = tabulate(elems, headers=headers, tablefmt="grid").split("\n")
    for row in table:
        print(row)


def print_total_rewards_table(length_rewards, quiz_rewards, total_rewards):
    advantages = compute_advantages(total_rewards)
    length = len(length_rewards)
    elems = list(zip(range(length), length_rewards, quiz_rewards, total_rewards, advantages))

    headers = ["Index", "Length Reward", "Quiz Reward", "Total Reward", "Advantage"]
    table = tabulate(elems, headers=headers, tablefmt="grid").split("\n")
    for row in table:
        print(row)