# bandit_env.py
import random
import verifiers as vf
from verifiers.envs.multiturn_env import MultiTurnEnv
from verifiers import RewardFunc   # base class
import re
from verifiers.utils.data_utils import extract_boxed_answer
from verifiers.parsers.parser import Parser
from datasets import Dataset
import numpy as np
from collections import Counter

SYS_PROMPT = """
Think step-by-step inside <think>...</think> tags.

Then, give your final numerical answer inside \\boxed{{...}}.
"""

def _parse_choice(text: str, n_arms: int) -> int | None:
    """
    Extracts the integer inside \\boxed{...}.  Returns None if the parse fails
    or the number is not in [0, n_arms-1].
    """
    # strip tags, keep only what’s inside \boxed{…}
    boxed = extract_boxed_answer(text)                # library helper :contentReference[oaicite:0]{index=0}
    # m = re.search(r"-?\d+", boxed)                    # first integer (handles whitespace, etc.)
    # if m is None:
    #     return None
    # val = int(m.group())
    # return val if 0 <= val < n_arms else None
    try:
        out=int(boxed)
        if out<0 or out>=n_arms: out=None
    except:
        out=None
    return out

def make_prompt_dataset(n: int, n_arms: int) -> Dataset:
    """
    Return a HuggingFace Dataset containing `n` identical examples:
        {"question": "Pick an integer between 0 and {n_arms-1} inclusive.", "answer": 0}

    Parameters
    ----------
    n : int
        Number of copies / rows in the dataset.
    n_arms : int
        Upper bound on the integer in the question prompt (exclusive).

    Returns
    -------
    datasets.Dataset
    """
    if n <= 0:
        raise ValueError("`n` must be positive.")
    if n_arms <= 0:
        raise ValueError("`n_arms` must be positive.")

    # Build the data once, then replicate `n` times.
    question = f"Your goal is to get a high reward over our conversation. In each of your responses, you will choose an integer between 0 and {n_arms-1}. Then you will be shown the reward for your choice. Each choice has a different reward and your goal is to get the highest reward over our conversation. Now let's start. Pick an integer between 0 and {n_arms - 1} inclusive."
    dataset_dict = {
        "question": [question] * n,
        "answer":   [0]        * n,
    }
    return Dataset.from_dict(dataset_dict)

def make_pick_integer_perm_dataset(n: int, n_arms: int, seed: int = 69) -> Dataset:
    """
    Return a HuggingFace Dataset with `n` rows:
       {"question": f"Pick an integer between 0 and {n_arms-1} inclusive.", 
        "answer":   <random permutation of range(n_arms)>}
    """
    if n <= 0 or n_arms <= 0:
        raise ValueError("`n` and `n_arms` must be positive integers")

    rng = np.random.default_rng(seed)          # fast, stateless PRNG
    #question = f"Pick an integer between 0 and {n_arms - 1} inclusive."
    question = f"Your goal is to get a high reward over our conversation. In each of your responses, you will choose an integer between 0 and {n_arms-1}. Then you will be shown the reward for your choice. Each choice has a different reward and your goal is to get the highest reward over our conversation. Now let's start. Pick an integer between 0 and {n_arms - 1} inclusive."
    answers = [rng.permutation(n_arms).tolist() for _ in range(n)]
    random.shuffle(answers)
    answer_counts = Counter(tuple(ans) for ans in answers)
    for ans_tuple, count in answer_counts.items():
        print(f"{list(ans_tuple)}: {count}")

    data = {
        "question": [question] * n,
        "answer":   answers,
    }
    return Dataset.from_dict(data)

class BoxedIntParser(vf.Parser):
    def __init__(self,
                 n_arms,
                 **kwargs):
        super().__init__(**kwargs)
        self.n_arms=n_arms

    def parse(self, response: str):
        boxed = extract_boxed_answer(response)
        # if boxed is None:
        #     raise ValueError("No boxed answer")
        # m = re.search(r"\d+", boxed)
        # if m is None:
        #     raise ValueError("No integer in boxed")
        # val = int(m.group())
        # if not (0 <= val < self.n_arms):  # Make n_arms accessible, e.g., pass in init
        #     raise ValueError("Out of range")
        # return val
        return boxed

    def get_format_reward_func(self):
        """
        Return a reward function that checks if each message follows the format:
        <think>
        ...
        </think>
        ...
        """
        def follows_format(text: str) -> float:
            if (
                text.strip().startswith("<think>") and 
                text.count("<think>") == 1 and
                text.count("</think>") == 1 and
                len(text.split("</think>")[-1]) > 0
            ):
                return 1.0
            return 0.0

        def format_reward_func(completion, **kwargs) -> float:
            messages = self.get_assistant_messages(completion)
            return sum(follows_format(m["content"]) for m in messages) / len(messages)
        return format_reward_func


class SumRewardVerifier(RewardFunc):
    """Rubric helper that just returns the trajectory reward stored in `state`."""
    def __call__(self, *, state, **_) -> float:      # **_ swallows unused kwargs
        return state["total_reward"]
    
def sum_reward_verifier(*, state, **_) -> float:
    return state["total_reward"]

def compute_rewards(completion,state):
    n_arms=state['n_arms']
    total_reward=0
    is_first=True
    for msg in completion:
        if msg['role'] == 'assistant':
            if is_first:
                is_first=False
                continue
            try:
                choice   = _parse_choice(msg['content'], n_arms)
            except Exception as e:
                print(f"Parse error in _parse_choice: {e!r}")
                choice=None

            if choice is None: total_reward-=1
            else: total_reward+=state['answer'][choice]

    return total_reward

class BanditEnv(MultiTurnEnv):
    def __init__(self, n_arms: int, n_turns: int, **kwargs):
        self.n_arms = n_arms
        super().__init__(
            # dummy dataset row; MultiTurnEnv expects something iterable
            dataset=make_pick_integer_perm_dataset(100000, n_arms),
            system_prompt=SYS_PROMPT,
            max_turns=n_turns+1,
            parser=Parser(),          # no tagging/formatting needed
            rubric=vf.Rubric([compute_rewards]), #vf.Rubric([sum_reward_verifier]),
            **kwargs
        )

    # ---------- helpers ----------------------------------------------------
    def _new_bandit(self):
        #return list(range(self.n_arms))
        return random.sample(range(self.n_arms), self.n_arms)  # random perm

    # ---------- required overrides -----------------------------------------
    def env_response(self, messages, state, **_):
        if 'turn_rewards' not in state:
            #state['arm_rewards']=self._new_bandit()
            state['step']=0
            state['total_reward']=0
            state['turn_rewards']=[]
            state['n_arms']=self.n_arms
        
        last_msg = messages[-1]["content"]
        try:
            choice   = _parse_choice(last_msg, self.n_arms)
        except Exception as e:
            print(f"Parse error in _parse_choice: {e!r}")
            choice=None

        if choice is None:                     # invalid → –1
            reward = -1
            print('-1 reward')
        else:                                  # valid → bandit payoff
            reward = state["answer"][choice]

        if len(state['turn_rewards'])>0: # try doing this to reduce variance
            state["total_reward"] += reward
        state["step"]         += 1
        state['turn_rewards'].append(reward)

        user_prompt = (f"Reward for your last response is {reward}. "
                       f"Pick an integer between 0 and {self.n_arms-1} inclusive.")

        return [{"role": "user", "content": user_prompt}], state

    def is_completed(self, messages, state, **_):
        if 'turn_rewards' not in state: return False
        #return len(state["turn_rewards"]) >= self.max_turns-1
        return len(state['responses'])>=self.max_turns-1
