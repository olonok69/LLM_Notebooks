import os
import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel
from random import shuffle
from openai import OpenAI
from tabulate import tabulate

load_dotenv()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
pb_client = OpenAI(
    base_url=os.environ["PREDIBASE_MODEL_LLAMA_URL"],
    api_key=os.environ["PREDIBASE_API_KEY"],
)


MODEL_NAME = "predibase/Meta-Llama-3.1-8B-Instruct-dequantized"


QUIZ_PROMPT = """Generate a multiple-choice quiz based on the information in the following earnings call transcript.

Example:

```
1. What was the q1 adjusted earnings per share?
a) $3.34
b) $5.32
c) $2.49
d) $7.78

2. By what percent did same store sales rise in q1?
a) 29.4%
b) 32.1%
c) 24.7%
d) 21.2%

===== ANSWERS =====
1. a
2. c
```

Limit the length of the quiz to the top 20 most relevant questions for financial analysts.

Transcript:

{text}
"""


class Question(BaseModel):
    text: str
    options: list[str]
    answer: int

    def shuffle_options(self) -> None:
        """Shuffle the options while preserving the correct answer"""
        # Get the correct answer text
        correct = self.options[self.answer]
        
        # Shuffle the options
        shuffled = self.options.copy()
        shuffle(shuffled)
        
        # Update the answer index to match new position
        self.options = shuffled
        self.answer = shuffled.index(correct)

    def __str__(self) -> str:
        """Pretty print a single question"""
        output = [self.text]
        for i, option in enumerate(self.options):
            output.append(f"{chr(65+i)}. {option}")
        return "\n".join(output)


class Quiz(BaseModel):
    questions: list[Question]

    def shuffle_all_questions(self) -> None:
        """Shuffle the options for all questions in the quiz"""
        for question in self.questions:
            question.shuffle_options()
    
    def __str__(self) -> str:
        """Pretty print the entire quiz"""
        output = []
        for i, question in enumerate(self.questions, 1):
            output.append(f"\nQuestion {i}:")
            output.append(str(question))
        return "\n".join(output)


letter_to_index = {"A": 0, "B": 1, "C": 2, "D": 3}
index_to_letter = ["A", "B", "C", "D"]


def take_quiz(summary: str, quiz: Quiz) -> list[str]:
    template = """Use the provided summary of a transcript to answer the following quiz.

Quiz:

{quiz}

Summary:

{summary}

Respond with just a list of answers and no additional text, for example:

[A, D, C, B, B, C, D, A, A, B]

You must provide an answer for all questions. If you don't know the answer, answer with "0" for that question. Example:

[A, D, 0, B, B, C, D, A, A, B]
"""

    question_strs = []
    for question in quiz.questions:
        question_str = question.text
        for i, option in enumerate(question.options):
            letter = index_to_letter[i]
            question_str += f"\n{letter}. {option}"
        question_strs.append(question_str)
    quiz_str = "\n\n".join(question_strs)

    prompt = template.format(quiz=quiz_str, summary=summary)
    # print(prompt)

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    resp_str = resp.choices[0].message.content
    
    # Convert string representation of list to actual list of strings
    answers = resp_str.strip('[]').split(', ')

    return answers


def generate_quiz(transcript: str) -> Quiz:
    prompt = QUIZ_PROMPT.format(text=transcript)
    messages = [
        {"role": "user", "content": prompt},
    ]
    resp = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7,
        response_format=Quiz,
    )
    quiz = resp.choices[0].message.parsed
    quiz.shuffle_all_questions()

    # Take quiz on transcript to identify answerable questions
    prev_len = len(quiz.questions)
    while True:
        answers = take_quiz(transcript, quiz)
        
        # Keep only questions where answer is correct on original transcript
        answerable_questions = []
        for answer, question in zip(answers, quiz.questions):
            expected_answer = index_to_letter[question.answer]
            if answer == expected_answer:
                answerable_questions.append(question)
                
        quiz.questions = answerable_questions
        
        # Break if no change in number of questions
        if len(quiz.questions) == prev_len:
            break
            
        prev_len = len(quiz.questions)
    
    # Limit to 10 questions
    quiz.questions = quiz.questions[:10]

    return quiz


def score_quiz_answers(answers: list[str], quiz: Quiz) -> float:
    assert len(answers) == len(quiz.questions)

    total = len(answers)
    correct = 0
    for answer, question in zip(answers, quiz.questions):
        expected_answer = index_to_letter[question.answer]
        if answer == expected_answer:
            correct += 1
    return correct / total


def quiz_reward(response: str, quiz: Quiz) -> float:
    answers = take_quiz(response, quiz)
    return score_quiz_answers(answers, quiz)


SUMMARIZE_PROMPT = """Generate a concise summary of the information in the following earnings call transcript.

Only respond with the summary, do not include any extraneous text.

Transcript:

{transcript}
"""


def summarize(transcript, n=1):
    prompt = SUMMARIZE_PROMPT.format(transcript=transcript)
    messages = [
        {"role": "user", "content": prompt},
    ]

    return pb_client.chat.completions.create(
        model="predibase/Meta-Llama-3.1-8B-Instruct-dequantized",
        messages=messages,
        n=n,
        temperature=0.9,
    )


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
    elems = list(zip(range(length), all_answers, rewards, advantages))

    headers = ["Index", "Answer", "Reward", "Advantage"]
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