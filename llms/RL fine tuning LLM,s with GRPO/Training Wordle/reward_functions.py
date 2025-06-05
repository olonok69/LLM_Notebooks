def output_format_check(prompt: str, completion: str, example: dict) -> int:
    import re
    import pandas as pd

    reward = 0
    try:
        # Add synthetic <think> as it's already part of the prompt and prefilled 
        # for the assistant to more easily match the regex
        completion = "<think>" + completion

        # Check if the format matches expected pattern:
        # <think> content </think> followed by <answer> content </answer>
        regex = (
            r"^<think>\s*([^<]*(?:<(?!/?think>)[^<]*)*)\s*<\/think>\n"
            r"<guess>\s*([\s\S]*?)\s*<\/guess>$"
        )

        # Search for the regex in the completion
        match = re.search(regex, completion, re.DOTALL)
        if match is None or len(match.groups()) != 2:
            return 0

        guess = match.groups()[1]
        guess = guess.strip()

        # If the word is not 5 characters, return 0
        if len(guess) != 5:
            return 0.1

        # Check if the guess is a valid word compared to a predifined list of words
        word_list = pd.read_csv(str(example["word_list"]))
        if guess not in word_list["Word"].values:
            return 0.5

        reward = 1.0
    except Exception:
        pass

    return reward


# Reward function that checks if the guess uses the previous feedback for its next guess
def uses_previous_feedback(prompt: str, completion: str, example: dict) -> int:
    import re
    import ast

    reward = 0
    try:
        # Add synthetic <think> as it's already part of the prompt and prefilled 
        # for the assistant to more easily match the regex
        completion = "<think>" + completion

        # Extract the guess from the completion
        regex = r"<guess>\s*([\s\S]*?)\s*<\/guess>$"
        match = re.search(regex, completion, re.DOTALL)
        if match is None or len(match.groups()) != 1:
            return 0

        guess = match.groups()[0].strip()
        if len(guess) != 5:
            return 0.0

        past_guess_history = ast.literal_eval(example["past_guess_history"])
        if len(past_guess_history) == 0:
            print("Uses previous feedback reward: 0.1 (No past guesses)")
            return 0.1

        correct_letter_to_position = {}
        valid_letter_to_position = {}
        wrong_letter_to_position = {}
        for _, past_feedback in past_guess_history:
            past_feedback = past_feedback.split(" ")
            for i, fb in enumerate(past_feedback):
                if '✓' in fb:
                    if fb[0] not in correct_letter_to_position:
                        correct_letter_to_position[fb[0]] = set()
                    correct_letter_to_position[fb[0]].add(i)
                elif '-' in fb:
                    if fb[0] not in valid_letter_to_position:
                        valid_letter_to_position[fb[0]] = set()
                    valid_letter_to_position[fb[0]].add(i)
                else:
                    if fb[0] not in wrong_letter_to_position:
                        wrong_letter_to_position[fb[0]] = set()
                    wrong_letter_to_position[fb[0]].add(i)

        for idx, letter in enumerate(guess):
            # Positive reward if guess reuses letter in confirmed correct position
            if (letter in correct_letter_to_position and idx in correct_letter_to_position[letter]):
                reward += 0.2
            # Reward if letter known to be in word is used in a new position
            elif (letter in valid_letter_to_position and idx not in valid_letter_to_position[letter]):
                reward += 0.1
            # Penalize reuse of known-in-word letter in same position (not exploring)
            elif (letter in valid_letter_to_position and idx in valid_letter_to_position[letter]):
                reward -= 0.2
            # Penalize use of known-absent letter
            elif letter in wrong_letter_to_position:
                reward -= 0.5
            else:
                # Reward unknown letters with partial credit for exploration
                reward += 0.05

    except Exception:
        return 0.0

    return reward


# Reward function that computes normalized information gain of the guess, i.e.,
# does the new guess reduce the uncertainty of the secret word the most
def guess_value(prompt: str, completion: str, example: dict) -> int:
    import math
    import re
    import ast
    import pandas as pd

    def validate_guess(secret: str, guess: str, raw_feedback: bool = False) -> str:
        feedback = []
        secret_list = list(secret)

        # Check for correct positions
        for i, (g_char, s_char) in enumerate(zip(guess, secret)):
            if g_char == s_char:
                feedback.append(f"{g_char}(✓) ")
                secret_list[i] = None
            else:
                feedback.append(None)

        # Check for misplaced letters
        for i, g_char in enumerate(guess):
            if feedback[i] is None:
                if g_char in secret_list:
                    feedback[i] = f"{g_char}(-) "
                    secret_list[secret_list.index(g_char)] = None
                else:
                    feedback[i] = f"{g_char}(x) "

        if raw_feedback:
            return feedback
        return "".join(feedback).strip()

    def filter_candidates(all_candidate_words, past_guesses):
        filtered = []
        for word in all_candidate_words:
            valid = True
            for past_guess, past_feedback in past_guesses:
                # Compute what the feedback would be if 'word' were the secret.
                candidate_feedback = validate_guess(word, past_guess)
                if candidate_feedback != past_feedback:
                    valid = False
                    break
            if valid:
                filtered.append(word)
        return filtered

    def compute_normalized_information_gain(all_candidate_words, past_guesses, guess):
        # First, filter the candidate words based on past guesses.
        candidates = filter_candidates(all_candidate_words, past_guesses)
        total_candidates = len(candidates)

        # If no candidates remain, return zeros.
        if total_candidates == 0:
            return 0.0, 0.0

        # Current uncertainty (entropy) before the guess.
        current_entropy = math.log2(total_candidates)

        # Partition candidates by the feedback pattern that would be produced by the current guess.
        feedback_groups = {}
        for word in candidates:
            # Get the raw feedback list (e.g., ['B(✓) ', 'R(✓) ', 'A(x) ', ...])
            feedback = validate_guess(word, guess, raw_feedback=True)
            # Create a simple representation for the feedback pattern.
            # '1' for correct position, '0' for wrong position, 'x' for letter not in word.
            feedback_pattern = "".join('1' if "✓" in fb else ('0' if "-" in fb else 'x') 
                                    for fb in feedback)
            feedback_groups.setdefault(feedback_pattern, []).append(word)

        expected_entropy = 0
        max_info_gain = 0
        # For each feedback group, compute its contribution to the expected entropy and the info gain.
        for group in feedback_groups.values():
            group_size = len(group)
            p = group_size / total_candidates
            # Entropy if this feedback is received.
            group_entropy = math.log2(group_size) if group_size > 0 else 0
            expected_entropy += p * group_entropy
            # Information gain for this feedback outcome.
            info_gain = current_entropy - group_entropy
            max_info_gain = max(max_info_gain, info_gain)

        # The expected gain is the reduction in entropy on average.
        expected_gain = current_entropy - expected_entropy

        # Normalize by the maximum possible gain, which is current_entropy (if you reduced to one candidate).
        normalized_expected_gain = expected_gain / current_entropy if current_entropy > 0 else 0
        normalized_max_gain = max_info_gain / current_entropy if current_entropy > 0 else 0

        return normalized_expected_gain, normalized_max_gain

    reward = 0
    try:
        # Add synthetic <think> as it's already part of the prompt and prefilled 
        # for the assistant to more easily match the regex
        completion = "<think>" + completion

        # Extract the guess from the completion
        regex = r"<guess>\s*([\s\S]*?)\s*<\/guess>$"
        match = re.search(regex, completion, re.DOTALL)
        if match is None or len(match.groups()) != 1:
            return 0

        guess = match.groups()[0].strip()
        if len(guess) != 5:
            return 0.0

        # Load the word list
        word_list = pd.read_csv(str(example["word_list"]))
        if guess not in word_list["Word"].values:
            return 0.0

        # Extract past guesses and feedback
        past_guess_history = ast.literal_eval(example["past_guess_history"])

        # Compute normalized information gain
        normalized_expected_gain, _ = compute_normalized_information_gain(
            word_list["Word"].values,
            past_guess_history,
            guess
        )

        # Compute reward based on normalized information gain
        reward = normalized_expected_gain
    except Exception:
        return 0.0

    return reward
