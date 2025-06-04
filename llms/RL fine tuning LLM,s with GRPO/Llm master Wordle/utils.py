import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import List
import numpy as np

from dotenv import load_dotenv
from openai import OpenAI
from transformers import AutoTokenizer
from tabulate import tabulate
from predibase import Predibase, DeploymentConfig


load_dotenv() 

base_model_id = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(base_model_id)


SYSTEM_PROMPT = """
You are playing Wordle, a word-guessing game.

### Game Rules:
- You have **6 tries** to guess a secret **5-letter** word.
- Each guess must be a valid **5-letter English word**.
- After each guess, you will receive feedback indicating how close your guess was.

### Feedback Format:
Each letter in your guess will receive one of three symbols:
1. ✓ : The letter is in the word and in the CORRECT position.
2. - : The letter is in the word but in the WRONG position.
3. x : The letter is NOT in the word.

### Example:
Secret Word: BRISK

Guess 1: STORM → Feedback: S(-) T(x) O(x) R(-) M(x)
Guess 2: BRAVE → Feedback: B(✓) R(✓) A(x) V(x) E(x)
Guess 3: BRISK → Feedback: B(✓) R(✓) I(✓) S(✓) K(✓)

### Response Format:
Think through the problem and feedback step by step. Make sure to first add your step by step thought process within <think> </think> tags. Then, return your guessed word in the following format: <guess> guessed-word </guess>.
"""


client = OpenAI(
    base_url=os.environ["PREDIBASE_MODEL_QWEN_URL"],
    api_key=os.environ["PREDIBASE_API_KEY"],
)

best_of_client = OpenAI(
    base_url=os.environ["PREDIBASE_MODEL_QWEN_URL"],
    api_key=os.environ["PREDIBASE_API_KEY"],
)


def generate_stream(
    prompt: str,
    adapter_id: str = "",
    temperature: float = 0.7,
    max_tokens: int = 1024,
    stream: bool = True,
) -> str:
    response = client.completions.create(
        model=adapter_id,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=stream,
    )

    completion = ""
    for chunk in response:
        if chunk.choices[0].text is not None:
            content = chunk.choices[0].text
            print(content, end="", flush=True)
            completion += content
    print()

    return completion


def generate(
    messages: List[dict],
    adapter_id: str = "",
    num_guesses: int = 1,
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> List[str]:
    if temperature > 0.0:
        completions = best_of_client.chat.completions.create(
            model=adapter_id,
            messages=messages,
            n=num_guesses,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return [choice.message.content for choice in completions.choices]
    else:
        return [
            best_of_client.chat.completions.create(
                model=adapter_id,
                messages=messages,
                n=1,
                temperature=temperature,
                max_tokens=max_tokens
            ).choices[0].message.content for _ in range(num_guesses)
        ]


def create_deployment(name: str = "qwen2-5-7b-instruct-dlai"):
    os.environ["PREDIBASE_GATEWAY"] = "https://api.staging.predibase.com"
    pb = Predibase(api_token=os.environ["PREDIBASE_API_KEY"])
    try:
        pb.deployments.create(
            name=name,
            config=DeploymentConfig(
                base_model="qwen2-5-7b-instruct",
                min_replicas=0,
                max_replicas=1,
                cooldown_time=1200,
                custom_args=[
                    "--max-best-of", "32",
                ]
            )
        )
    except Exception:
        print(f"Deployment {name} already exists")


class LetterFeedback(Enum):
    CORRECT = "✓"
    WRONG_POS = "-"
    WRONG_LETTER = "x"


def get_feedback(guess: str, secret_word: str) -> List[LetterFeedback]:
    valid_letters = set(secret_word)
    feedback = []
    for letter, secret_letter in zip(guess, secret_word):
        if letter == secret_letter:
            feedback.append(LetterFeedback.CORRECT)
        elif letter in valid_letters:
            feedback.append(LetterFeedback.WRONG_POS)
        else:
            feedback.append(LetterFeedback.WRONG_LETTER)
    return feedback


@dataclass
class GuessWithFeedback:
    guess: str
    feedback: List[LetterFeedback]

    def __repr__(self) -> str:
        feedback_str = " ".join(f"{letter}({fb.value})" for letter, fb in zip(self.guess, self.feedback))
        return f"{self.guess} → Feedback: {feedback_str}"

    @staticmethod
    def from_secret(guess: str, secret: str) -> "GuessWithFeedback":
        return GuessWithFeedback(guess, get_feedback(guess, secret))


def render_user_prompt(past_guesses: List[GuessWithFeedback]) -> str:
    prompt = "Make a new 5-letter word guess."
    if past_guesses:
        prompt += "\n\nHere is some previous feedback:"
        for i, guess in enumerate(past_guesses):
            prompt += f"\nGuess {i+1}: {guess}"
    return prompt


def get_messages(past_guesses: List[GuessWithFeedback]):
    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": render_user_prompt(past_guesses)
        },
        {
            "role": "assistant",
            "content": "Let me solve this step by step.\n<think>"
        }
    ]


def render_prompt(past_guesses: List[GuessWithFeedback]):
    messages = get_messages(past_guesses)
    return tokenizer.apply_chat_template(
        messages, tokenize=False, continue_final_message=True
    )


def extract_guess(completion: str) -> str:
    match = re.search(r"<guess>\s*([\s\S]*?)\s*<\/guess>", completion, re.DOTALL)
    if not match:
        return ""
    return match.group(1).strip().upper()


def next_turn(past_guesses: List[GuessWithFeedback], secret_word: str, adapter_id = ""):
    prompt = render_prompt(past_guesses)
    completion = generate_stream(prompt)
    guess = extract_guess(completion)

    feedback = get_feedback(guess, secret_word)
    past_guesses.append(GuessWithFeedback(guess, feedback))
    print("\n\n")
    print(("-" * 100) + "\n")
    for past_guess in past_guesses:
        print(past_guess)

    if guess == secret_word:
        print("🎉 SUCCESS 🎉")
    elif len(past_guesses) >= 6:
        print("❌ better luck next time... ❌")


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


def print_guesses_table(extracted_guesses, rewards):
    advantages = compute_advantages(rewards)
    length = len(extracted_guesses)
    elems = list(zip(range(length), extracted_guesses, rewards, advantages))

    headers = ["Index", "Guess", "Reward", "Advantage"]
    table = tabulate(elems, headers=headers, tablefmt="grid").split("\n")
    for row in table:
        print(row)
