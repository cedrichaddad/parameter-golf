import modal
import sys
import subprocess
import os
import glob
import shutil
from collections import deque

app = modal.App("pg-train-detached")

image = (
    modal.Image.from_dockerfile("deploy/Dockerfile", context_dir=".", add_python="3.12")
    .pip_install("huggingface_hub")
)

data_volume = modal.Volume.from_name("pg-data", create_if_missing=True)
output_volume = modal.Volume.from_name("pg-output", create_if_missing=True)


def _maybe_seed_data_env():
    if os.environ.get("PG_TRAIN_GLOB") and os.environ.get("PG_VAL_GLOB"):
        return

    candidates = [
        os.environ.get("DATA_DIR"),
        "/data/datasets/fineweb10B_sp8192",
        "/data/datasets/fineweb10B_sp1024",
    ]
    for root in candidates:
        if not root:
            continue
        train_glob = os.path.join(root, "fineweb_train_*.bin")
        val_glob = os.path.join(root, "fineweb_val_*.bin")
        if not os.environ.get("PG_TRAIN_GLOB") and glob.glob(train_glob):
            os.environ["PG_TRAIN_GLOB"] = train_glob
        if not os.environ.get("PG_VAL_GLOB") and glob.glob(val_glob):
            os.environ["PG_VAL_GLOB"] = val_glob
        if "sp8192" in root and not os.environ.get("PG_TOKENIZER_VOCAB"):
            for vocab_path in (
                os.path.join(root, "tokenizer.vocab"),
                "/data/tokenizers/fineweb_8192_bpe.vocab",
            ):
                if os.path.exists(vocab_path):
                    os.environ["PG_TOKENIZER_VOCAB"] = vocab_path
                    break
        if os.environ.get("PG_TRAIN_GLOB") and os.environ.get("PG_VAL_GLOB"):
            break


def _run_pg_train(args: list[str], label: str):
    os.environ["RUST_LOG"] = "info"
    os.environ.setdefault("DATA_DIR", "/data/datasets/fineweb10B_sp8192")
    _maybe_seed_data_env()
    forwarded = list(args)
    mode = "smoke"
    if "--mode" in forwarded:
        mode_idx = forwarded.index("--mode")
        if mode_idx + 1 < len(forwarded):
            mode = forwarded[mode_idx + 1]
    if os.environ.get("PG_TRAIN_GLOB") and "--train-data" not in forwarded:
        forwarded.extend(["--train-data", os.environ["PG_TRAIN_GLOB"]])
    include_val_data = os.environ.get("PG_INCLUDE_VAL_DATA") == "1"
    if include_val_data and os.environ.get("PG_VAL_GLOB") and "--val-data" not in forwarded:
        forwarded.extend(["--val-data", os.environ["PG_VAL_GLOB"]])
    if os.environ.get("PG_TOKENIZER_VOCAB") and "--tokenizer-vocab" not in forwarded:
        forwarded.extend(["--tokenizer-vocab", os.environ["PG_TOKENIZER_VOCAB"]])
    if mode != "record" and "--eval-max-tokens" not in forwarded:
        forwarded.extend(["--eval-max-tokens", os.environ.get("PG_EVAL_MAX_TOKENS", "16384")])
    cmd = ["pg-train"] + forwarded
    print(f"Running {label} command:", " ".join(cmd), flush=True)
    print(
        "Data environment:",
        {
            "DATA_DIR": os.environ.get("DATA_DIR"),
            "PG_TRAIN_GLOB": os.environ.get("PG_TRAIN_GLOB"),
            "PG_VAL_GLOB": os.environ.get("PG_VAL_GLOB"),
            "PG_TOKENIZER_VOCAB": os.environ.get("PG_TOKENIZER_VOCAB"),
        },
        flush=True,
    )

    tail = deque(maxlen=400)
    proc = subprocess.Popen(
        cmd,
        env=os.environ,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc.stdout is None:
        raise RuntimeError("subprocess stdout pipe was not created")
    for line in iter(proc.stdout.readline, ""):
        tail.append(line)
        print(line, end="", flush=True)

    proc.wait()
    result = {
        "command": cmd,
        "returncode": proc.returncode,
        "tail": "".join(tail),
    }
    if proc.returncode != 0:
        raise RuntimeError(
            f"{label} command failed with code {proc.returncode}\n"
            f"Command: {' '.join(cmd)}\n"
            f"Last output:\n{result['tail']}"
        )
    output_volume.commit()
    return result


def _run_pg_eval(args: list[str]):
    os.environ["RUST_LOG"] = "info"
    os.environ.setdefault("DATA_DIR", "/data/datasets/fineweb10B_sp8192")
    _maybe_seed_data_env()
    forwarded = list(args)
    if os.environ.get("PG_VAL_GLOB") and "--val-data" not in forwarded:
        forwarded.extend(["--val-data", os.environ["PG_VAL_GLOB"]])
    if os.environ.get("PG_TOKENIZER_VOCAB") and "--tokenizer-vocab" not in forwarded:
        forwarded.extend(["--tokenizer-vocab", os.environ["PG_TOKENIZER_VOCAB"]])
    if os.environ.get("PG_EVAL_MAX_TOKENS") and "--max-tokens" not in forwarded:
        forwarded.extend(["--max-tokens", os.environ["PG_EVAL_MAX_TOKENS"]])
    cmd = ["pg-eval"] + forwarded
    print("Running eval command:", " ".join(cmd), flush=True)

    tail = deque(maxlen=400)
    proc = subprocess.Popen(
        cmd,
        env=os.environ,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc.stdout is None:
        raise RuntimeError("subprocess stdout pipe was not created")
    for line in iter(proc.stdout.readline, ""):
        tail.append(line)
        print(line, end="", flush=True)
    proc.wait()
    result = {
        "command": cmd,
        "returncode": proc.returncode,
        "tail": "".join(tail),
    }
    if proc.returncode != 0:
        raise RuntimeError(
            f"eval command failed with code {proc.returncode}\n"
            f"Command: {' '.join(cmd)}\n"
            f"Last output:\n{result['tail']}"
        )
    return result

def _run_pg_bench(args: list[str]):
    os.environ["RUST_LOG"] = "info"
    if not args:
        raise RuntimeError("bench requires a binary name")
    allowed = {
        "parity-kernels": "pg-parity-kernels",
        "parity-forward": "pg-parity-forward",
        "parity-step": "pg-parity-step",
        "gemm-bench": "pg-gemm-bench",
        "nccl-bench": "pg-nccl-bench",
        "preliminary": "pg-preliminary",
        "smoke": "pg-smoke",
    }
    binary = allowed.get(args[0])
    if binary is None:
        raise RuntimeError(f"unsupported bench binary {args[0]!r}; allowed={sorted(allowed)}")
    cmd = [binary] + list(args[1:])
    print("Running bench command:", " ".join(cmd), flush=True)

    tail = deque(maxlen=400)
    proc = subprocess.Popen(
        cmd,
        env=os.environ,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc.stdout is None:
        raise RuntimeError("subprocess stdout pipe was not created")
    for line in iter(proc.stdout.readline, ""):
        tail.append(line)
        print(line, end="", flush=True)
    proc.wait()
    result = {
        "command": cmd,
        "returncode": proc.returncode,
        "tail": "".join(tail),
    }
    if proc.returncode != 0:
        raise RuntimeError(
            f"bench command failed with code {proc.returncode}\n"
            f"Command: {' '.join(cmd)}\n"
            f"Last output:\n{result['tail']}"
        )
    return result


@app.function(
    image=image,
    timeout=3600,
    startup_timeout=900,
    volumes={
        "/data": data_volume,
    },
)
def seed_data():
    from huggingface_hub import snapshot_download

    dataset_dir = "/data/datasets/fineweb10B_sp8192"
    train_files = sorted(glob.glob(os.path.join(dataset_dir, "fineweb_train_*.bin")))
    val_files = sorted(glob.glob(os.path.join(dataset_dir, "fineweb_val_*.bin")))
    vocab_path = "/data/tokenizers/fineweb_8192_bpe.vocab"
    if train_files and val_files and os.path.exists(vocab_path):
        print(
            "SP8192 data already present:",
            {
                "train_files": len(train_files),
                "val_files": len(val_files),
                "vocab_path": vocab_path,
            },
            flush=True,
        )
    else:
        print("Downloading SP8192 shards/tokenizer into pg-data volume", flush=True)
        snapshot_download(
            repo_id="sproos/parameter-golf-tokenizers",
            local_dir="/data",
            allow_patterns=[
                "datasets/fineweb10B_sp8192/*",
                "tokenizers/fineweb_8192_bpe.model",
                "tokenizers/fineweb_8192_bpe.vocab",
            ],
        )

    os.makedirs(dataset_dir, exist_ok=True)
    if os.path.exists(vocab_path):
        shutil.copyfile(vocab_path, os.path.join(dataset_dir, "tokenizer.vocab"))
    train_files = sorted(glob.glob(os.path.join(dataset_dir, "fineweb_train_*.bin")))
    val_files = sorted(glob.glob(os.path.join(dataset_dir, "fineweb_val_*.bin")))
    result = {
        "dataset_dir": dataset_dir,
        "train_files": len(train_files),
        "val_files": len(val_files),
        "tokenizer_vocab": vocab_path if os.path.exists(vocab_path) else None,
    }
    print("Seed-data result:", result, flush=True)
    if not train_files or not val_files or not result["tokenizer_vocab"]:
        raise RuntimeError(f"SP8192 seed incomplete: {result}")
    data_volume.commit()
    return result

@app.function(
    image=image,
    gpu="H100:1",
    timeout=3600,
    startup_timeout=900,
    volumes={
        "/data": data_volume,
        "/output": output_volume,
    },
)
def run_command(args: list[str]):
    return _run_pg_train(args, "single-GPU")


@app.function(
    image=image,
    gpu="H100:8",
    timeout=3600,
    startup_timeout=900,
    volumes={
        "/data": data_volume,
        "/output": output_volume,
    },
)
def run_command_multi(args: list[str]):
    return _run_pg_train(args, "multi-GPU")


@app.function(
    image=image,
    gpu="H100:8",
    timeout=1800,
    startup_timeout=900,
    volumes={
        "/data": data_volume,
        "/output": output_volume,
    },
)
def run_eval_command(args: list[str]):
    return _run_pg_eval(args)

@app.function(
    image=image,
    gpu="H100:1",
    timeout=1800,
    startup_timeout=900,
    volumes={
        "/data": data_volume,
        "/output": output_volume,
    },
)
def run_bench_command(args: list[str]):
    return _run_pg_bench(args)

@app.local_entrypoint()
def main(*args: str):
    use_multi = False
    forwarded = list(args)
    wait_for_result = os.environ.get("PG_WAIT") == "1"
    if forwarded and forwarded[0] == "seed-data":
        if wait_for_result:
            result = seed_data.remote()
            print("Seed-data result:", result, flush=True)
            return
        call = seed_data.spawn()
        call_id = getattr(call, "object_id", None) or getattr(call, "id", None)
        print("Spawned seed-data Modal call:", call_id or call, flush=True)
        return
    if forwarded and forwarded[0] == "eval":
        if wait_for_result:
            result = run_eval_command.remote(forwarded[1:])
            print("Eval result:", result, flush=True)
            return
        call = run_eval_command.spawn(forwarded[1:])
        call_id = getattr(call, "object_id", None) or getattr(call, "id", None)
        print("Spawned eval Modal call:", call_id or call, flush=True)
        return
    if forwarded and forwarded[0] == "bench":
        if wait_for_result:
            result = run_bench_command.remote(forwarded[1:])
            print("Bench result:", result, flush=True)
            return
        call = run_bench_command.spawn(forwarded[1:])
        call_id = getattr(call, "object_id", None) or getattr(call, "id", None)
        print("Spawned bench Modal call:", call_id or call, flush=True)
        return
    if forwarded and forwarded[0] == "--multi":
        use_multi = True
        forwarded = forwarded[1:]
    if os.environ.get("PG_MULTI_GPU") == "1":
        use_multi = True
    print(
        "Dispatching command to detached runner:",
        {"multi_gpu": use_multi, "args": forwarded},
        flush=True,
    )
    if use_multi:
        if wait_for_result:
            result = run_command_multi.remote(forwarded)
            print("Multi-GPU result:", result, flush=True)
            return
        call = run_command_multi.spawn(forwarded)
    else:
        if wait_for_result:
            result = run_command.remote(forwarded)
            print("Single-GPU result:", result, flush=True)
            return
        call = run_command.spawn(forwarded)
    call_id = getattr(call, "object_id", None) or getattr(call, "id", None)
    print("Spawned Modal call:", call_id or call, flush=True)
