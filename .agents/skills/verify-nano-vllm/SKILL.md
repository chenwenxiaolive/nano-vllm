---
name: verify-nano-vllm
description: "Runs nano-vllm on a GPU node for proof: builds the environment, drives the example, the benchmark and tensor parallelism, and captures the result. Use when a change to nano-vllm needs an acceptance run, or a review has to launch it."
---

# Verify nano-vllm

nano-vllm is a library with two entry scripts and no server. A run loads the
model, generates and exits, so launching and driving a feature are one command
and nothing stays up between runs.

Run every command below from the node checkout, in one bash shell. They are
written for a `zg` node.

## Environment

No GPU. Once per checkout:

```bash
uv venv --seed --system-site-packages --python /usr/bin/python3 .venv
.venv/bin/python -m pip install --no-input transformers xxhash
```

- The node image's Python holds torch (NVIDIA's build), flash-attn and triton. The environment sits on those and adds the two packages the image lacks. Nothing installs `nanovllm`: the scripts import it from the checkout.
- `pyproject.toml` has no lock, so transformers is whatever is current that day.
- The model is Qwen3-0.6B, on zg at `/volume/tmp/nano-vllm-model-20261114-vCIsqc/Qwen3-0.6B`. Where that is missing, download it (1.5 GB) and use the new path instead:

  ```bash
  MODEL=$(mktemp -d /volume/tmp/nano-vllm-model-$(date +%Y%m%d)-XXXXXX)
  HF_HOME=$MODEL/hf_home .venv/bin/hf download Qwen/Qwen3-0.6B --local-dir $MODEL/Qwen3-0.6B
  ```

## Check

```bash
git status --short --branch; git rev-parse HEAD; .venv/bin/python -c "import torch, flash_attn, transformers, xxhash; print(torch.__version__, flash_attn.__version__, transformers.__version__)"; nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader; ls /dev/shm/nanovllm
```

Worth driving when the status lists no file, the commit is the one under test,
the import prints three versions, the GPUs to use hold nothing but keepalive
workers and `/dev/shm/nanovllm` does not exist. Every run binds
`tcp://localhost:2333` and a tensor-parallel one creates that shared memory, so
a node takes one nano-vllm run at a time.

## Run directory

```bash
RUN=$(mktemp -d /volume/tmp/nano-vllm-verify-$(date +%Y%m%d)-XXXXXX)
mkdir -p $RUN/home/huggingface
ln -s /volume/tmp/nano-vllm-model-20261114-vCIsqc/Qwen3-0.6B $RUN/home/huggingface/Qwen3-0.6B
git rev-parse HEAD > $RUN/commit.txt
.venv/bin/python -m pip freeze > $RUN/versions.txt
export HOME=$RUN/home PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_DEVICE_ORDER=PCI_BUS_ID
export CPATH=/usr/local/cuda/include TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
drive() { name=$1; shift; "$@" > $RUN/$name.out 2>&1 & echo $! > $RUN/$name.pid; wait $!; echo "$name exit $?" >> $RUN/exit.txt; }
```

- `example.py` and `bench.py` read the model from `~/huggingface/Qwen3-0.6B/`, so `HOME` is a directory holding that path as a link.
- The image's triton has neither `cuda.h` nor `ptxas`. Without `CPATH` and `TRITON_PTXAS_PATH`, torch.compile ends with `fatal error: cuda.h: No such file or directory` or `Cannot find ptxas`.

## Drive

Each feature needs GPUs, so these runs belong to the session that holds the
node's allocation. One GPU is enough for the first two; a run takes 90% of its
memory.

| Feature | Command | Proof |
|---|---|---|
| Chat generation, eager | `CUDA_VISIBLE_DEVICES=0 drive example .venv/bin/python example.py` | `example exit 0` in `$RUN/exit.txt`; `grep -c '^Completion: ' $RUN/example.out` prints 2 and both completions are English text on their prompt |
| Batch of 256 sequences, compiled with CUDA graphs | `CUDA_VISIBLE_DEVICES=0 drive bench .venv/bin/python bench.py` | `bench exit 0`; the last line of `$RUN/bench.out` starts `Total: 133966tok`, the count the README's table gives |
| Tensor parallelism over two GPUs | `CUDA_VISIBLE_DEVICES=0,1 drive tp2 .venv/bin/python $RUN/tp2.py`, after writing the script below | `tp2 exit 0`; the `Completion:` line in `$RUN/tp2.out` names Paris |

```bash
cat > $RUN/tp2.py <<'EOF'
import os

from nanovllm import LLM, SamplingParams


def main():
    llm = LLM(os.path.expanduser("~/huggingface/Qwen3-0.6B/"), tensor_parallel_size=2)
    outputs = llm.generate(["The capital of France is"], SamplingParams(temperature=0.6, max_tokens=32))
    print(f"Completion: {outputs[0]['text']!r}")


if __name__ == "__main__":
    main()
EOF
```

The example takes about a minute, the benchmark about two, most of it
compiling. The benchmark printed 7976 tok/s on one H20-3e (one run,
2026-11-14): a size to expect, not a baseline to compare a change against.

Without a GPU a layer can still be built, which is as far as a CPU-only
verifier gets. `Qwen3Attention` reads the world size, so give it a one-process
group first:

```python
import torch.distributed as dist
dist.init_process_group("gloo", store=dist.HashStore(), rank=0, world_size=1)
```

## Evidence

`$RUN` holds `commit.txt`, `versions.txt`, `exit.txt` and one `.out` and `.pid`
per feature. Keep the directory and name it where the result is reported.

## Cleanup

A run exits on its own. Stop one that hangs with `kill $(cat $RUN/<feature>.pid)`;
the tensor-parallel workers end with it. Then nothing may be left running from
the checkout:

```bash
for p in /proc/[0-9]*; do [ "$(readlink $p/cwd)" = "$(pwd -P)" ] && [ ${p#/proc/} != $$ ] && ps -o pid=,cmd= -p ${p#/proc/}; done
```

It prints nothing when the checkout is idle. The environment and the model stay.
