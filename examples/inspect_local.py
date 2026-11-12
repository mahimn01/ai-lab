"""Example eval: score a local model on 20 maths questions with Inspect AI, through the lab's server.

What it does
  Asks the model 20 questions from the public MATH-500 set and has Inspect check
  whether each reply contains the right final answer. Inspect sends the questions in
  parallel; the local server batches them on the GPU, which is where the speed comes from.

Run (two terminals, both from ~/Documents/Dev/ai-lab):
  1. Start the server:
       uv run mlx_lm.server --model mlx-community/Qwen3.5-4B-4bit --port 8642 \\
           --decode-concurrency 32 --prompt-concurrency 8 \\
           --chat-template-args '{"enable_thinking": false}'
  2. Run the eval:
       MLX_BASE_URL=http://127.0.0.1:8642/v1 MLX_API_KEY=local \\
       uv run --group evals inspect eval examples/inspect_local.py \\
           --model openai-api/mlx/mlx-community/Qwen3.5-4B-4bit --max-connections 32 \\
           --log-dir runs/inspect-logs

Checked: 2026-11-11, 20 samples in 57 s, accuracy 0.70.

Changes: Inspect writes one log file under runs/inspect-logs/. Nothing is sent outside this Mac
         (the "API key" is a placeholder; the server ignores it).
"""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import includes
from inspect_ai.solver import generate, system_message

SYSTEM = ("Solve the problem. Keep the working short and finish with the final answer "
          "on its own line in the form 'Answer: <answer>'.")


@task
def math20() -> Task:
    dataset = hf_dataset("HuggingFaceH4/MATH-500", split="test", limit=20,
                         sample_fields=lambda r: Sample(input=r["problem"], target=r["answer"]))
    return Task(dataset=dataset, solver=[system_message(SYSTEM), generate()], scorer=includes())
