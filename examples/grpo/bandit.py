import verifiers as vf
from verifiers.envs.bandit_env import BanditEnv

"""
first time:
import nltk
nltk.download('words', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

inference:
CUDA_VISIBLE_DEVICES=0 NCCL_P2P_DISABLE=1 vf-vllm --model willcb/Qwen3-1.7B
training:
CUDA_VISIBLE_DEVICES=1 NCCL_P2P_DISABLE=1 accelerate launch --num-processes 1 --config-file configs/zero3.yaml verifiers/examples/bandit.py
"""

model_name = "willcb/Qwen3-1.7B" #'willcb/Qwen3-0.6B' #"Qwen/Qwen2.5-1.5B-Instruct"
model, tokenizer = vf.get_model_and_tokenizer(model_name)

vf_env = BanditEnv(
    n_arms=2,
    n_turns=2
)

run_name = f"Bandit"
training_args=vf.grpo_defaults(run_name=run_name)
training_args.num_iterations=1
training_args.per_device_train_batch_size=4
training_args.num_generations=2
training_args.gradient_accumulation_steps=128
training_args.max_prompt_length=1024
training_args.max_completion_length=1024*2
training_args.max_steps=100000
training_args.mask_env_responses=False
training_args.warmup_steps=4
training_args.logging_steps=1
training_args.beta=0.0
training_args.async_generation_timeout=1000
training_args.temperature=1.2
training_args.learning_rate=1e-6
training_args.loss_type='sft'

trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=training_args,
)
trainer.train()
