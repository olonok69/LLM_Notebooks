#!/usr/bin/env python
# coding: utf-8

# # Lesson 3: Can a large language model master Wordle?

# <div style="background-color:#fff6ff; padding:13px; border-width:3px; border-color:#efe6ef; border-style:solid; border-radius:6px">
# <p> 💻 &nbsp; <b>Access <code>requirements.txt</code>  file:</b> 1) click on the <em>"File"</em> option on the top menu of the notebook and then 2) click on <em>"Open"</em>.
# 
# <p> ⬇ &nbsp; <b>Download Notebooks:</b> 1) click on the <em>"File"</em> option on the top menu of the notebook and then 2) click on <em>"Download as"</em> and select <em>"Notebook (.ipynb)"</em>.</p>
# 
# <p> 📒 &nbsp; For more help, please see the <em>"Appendix – Tips, Help, and Download"</em> Lesson.</p>
# 
# </div>

# <p style="background-color:#f7fff8; padding:15px; border-width:3px; border-color:#e0f0e0; border-style:solid; border-radius:6px"> 🚨
# &nbsp; <b>Different Run Results:</b> The output generated by AI chat models can vary with each execution due to their dynamic, probabilistic nature. Don't be surprised if your results differ from those shown in the video.</p>

# Start by load dependencies and setting up the Predibase client, which you'll use to call both base and finetuned models:

# In[ ]:


import os

from dotenv import load_dotenv
from openai import OpenAI

# Initialize client to query Qwen2.5 7B Instruct on Predibase using the OpenAI client.

_ = load_dotenv()

client = OpenAI(
    base_url=os.environ["PREDIBASE_MODEL_QWEN_URL"], # Qwen 2.5 7B Instruct
    api_key=os.environ["PREDIBASE_API_KEY"],
)


# Load the model tokenizer from Hugging face:

# In[ ]:


from transformers import AutoTokenizer

base_model_id = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(base_model_id)


# ## Setting up prompts to play Wordle

# In[ ]:


SYSTEM_PROMPT = """
You are playing Wordle, a word-guessing game.

### Game Rules:
- You have **6 tries** to guess a secret **5-letter** word.
- Each guess must be a valid **5-letter English word**.
- After each guess, you will receive feedback indicating how close 
your guess was.

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
Think through the problem and feedback step by step. Make sure to 
first add your step by step thought process within <think> </think> 
tags. Then, return your guessed word in the following format: 
<guess> guessed-word </guess>.
"""


# In[ ]:


from dataclasses import dataclass
from enum import Enum
from typing import List


class LetterFeedback(Enum):
    CORRECT = "✓"
    WRONG_POS = "-"
    WRONG_LETTER = "x"


@dataclass
class GuessWithFeedback:
    guess: str
    feedback: List[LetterFeedback]

    def __repr__(self) -> str:
        """Returns a readable string showing the guess alongside
        its letter-by-letter feedback."""
        feedback_str = " ".join(f"{letter}({fb.value})" for letter, fb in zip(self.guess, self.feedback))
        return f"{self.guess} → Feedback: {feedback_str}"


# In[ ]:


def render_user_prompt(past_guesses: List[GuessWithFeedback]) -> str:
    """Creates a user-facing prompt that includes past guesses 
    and their feedback."""
    prompt = "Make a new 5-letter word guess."
    if past_guesses:
        prompt += "\n\nHere is some previous feedback:"
        for i, guess in enumerate(past_guesses):
            prompt += f"\nGuess {i+1}: {guess}"
    return prompt


# In[ ]:


def render_prompt(past_guesses: List[GuessWithFeedback]):
    """Formats a full chat prompt using a system message, user 
    prompt, and assistant preamble to start the model's 
    step-by-step reasoning."""
    messages = [
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

    return tokenizer.apply_chat_template(
        messages, tokenize=False, continue_final_message=True
    )


# In[ ]:


def generate_stream(prompt: str, adapter_id: str = "") -> str:
    """Streams a model-generated response from a prompt in 
    real-time and prints it as it arrives."""
    response = client.completions.create(
        model=adapter_id,
        prompt=prompt,
        # Produce deterministic responses for evaluation
        temperature=0.0, 
        max_tokens=2048,
        stream=True,
    )
    
    completion = ""
    for chunk in response:
        if chunk.choices[0].text is not None:
            content = chunk.choices[0].text
            print(content, end="", flush=True)
            completion += content
    print()

    return completion


# ## Play a single turn of Wordle
# 
# Start by setting up the prompt with feedback for two prior guesses:

# In[ ]:


secret_word = "CRAFT"

past_guesses = [
    GuessWithFeedback(
        "CRANE", [
            LetterFeedback.CORRECT, 
            LetterFeedback.CORRECT, 
            LetterFeedback.CORRECT, 
            LetterFeedback.WRONG_LETTER, 
            LetterFeedback.WRONG_LETTER,
        ]),
    GuessWithFeedback(
        "CRASH", [
            LetterFeedback.CORRECT, 
            LetterFeedback.CORRECT, 
            LetterFeedback.CORRECT, 
            LetterFeedback.WRONG_LETTER, 
            LetterFeedback.WRONG_LETTER,
        ]),
]

prompt = render_prompt(past_guesses)
print(prompt)


# Prompt the base model and examine its response:

# In[ ]:


base_completion = generate_stream(prompt)


# Now prompt the RFT model and see how it responds:

# <div style="background-color:#fff6ff; padding:13px; border-width:3px; border-color:#efe6ef; border-style:solid; border-radius:6px"> <b>NOTE:</b> The cell below loads a model that has been tuned using RFT to play wordle. You'll learn how this model was trained throughout the course, and see the exact setup used to train the model on the Predibase platform in L8.
# </div>

# In[ ]:


# prompt fine-tuned model
ft_completion = generate_stream(prompt, adapter_id="wordle-dlai/2")


# ## Playing a game of wordle
# 
# Start by setting up a helper function that gets feedback on a guess:

# In[ ]:


import re

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


# Now create a `next_turn` function that incorporates feedback on a guess into the prompt to the LLM:

# In[ ]:


def next_turn(
    past_guesses: List[GuessWithFeedback], 
    secret_word: str, 
    adapter_id = ""
):
    prompt = render_prompt(past_guesses)
    completion = generate_stream(prompt, adapter_id)
    match = re.search(
        r"<guess>\s*(.*?)\s*</guess>", completion, re.DOTALL
    )
    if not match:
        raise RuntimeError("invalid guess")
    
    guess = match.group(1).upper()
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


# Try playing with the base model:

# In[ ]:


secret_word = "BRICK"
past_guesses = []
adapter_id = ""


# In[ ]:


next_turn(past_guesses, secret_word, adapter_id)


# Now try with the finetuned model:

# In[ ]:


secret_word = "BRICK"
past_guesses = []
adapter_id = "wordle-dlai/2"


# In[ ]:


next_turn(past_guesses, secret_word, adapter_id)


# ## Try for yourself!
# 
# Try different secret words above and see how the model responds. 
