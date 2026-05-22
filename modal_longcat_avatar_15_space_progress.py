"""Run LongCat-Video-Avatar-1.5 on Modal using real DramaBox stems.

Goal: get a bounded first proof of MindExpander + NASA/Apollo radio voices
speaking about space progress on the moon.
"""
from __future__ import annotations

import modal

REPO_ROOT = "/opt/data/workspace/github-forks/LongCat-Video-Avatar"
REMOTE_REPO = "/workspace/LongCat-Video"

app = modal.App("longcat-avatar-15-space-progress")

hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
longcat_models = modal.Volume.from_name("longcat-avatar-models", create_if_missing=True)
dramabox_outputs = modal.Volume.from_name("dramabox-outputs", create_if_missing=True)
longcat_outputs = modal.Volume.from_name("longcat-avatar-outputs", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "ffmpeg", "libsndfile1", "libgl1", "libglib2.0-0", "ninja-build", "curl")
    .run_commands(
        "python -m pip install --upgrade pip setuptools wheel packaging psutil ninja",
        "pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124",
        "pip install numpy==1.26.4 transformers==4.41.0 diffusers==0.35.1 einops==0.8.0 ftfy==6.2.0 psutil==6.0.0 av==12.0.0 opencv-python-headless==4.9.0.80 imageio==2.37.0 imageio-ffmpeg==0.6.0",
        "pip install scikit-learn==1.6.1 scikit-image==0.25.2 scipy==1.15.3 soundfile==0.13.1 soxr==0.5.0.post1 librosa==0.11.0 sympy==1.13.1 audio-separator==0.30.2 pyloudnorm==0.1.1 nvidia-ml-py==13.580.65 onnx==1.18.0 onnxruntime==1.16.3",
        "pip install huggingface_hub[hf_transfer]>=0.25.0 pillow loguru",
        "pip install flash-attn==2.7.4.post1 --no-build-isolation",
    )
    .env({
        "PYTHONUNBUFFERED": "1",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HOME": "/hf-cache",
        "HF_HUB_CACHE": "/hf-cache/hub",
    })
    .add_local_dir(REPO_ROOT, REMOTE_REPO)
)


@app.function(
    image=image,
    gpu="H100",
    memory=65536,
    timeout=7200,
    volumes={
        "/hf-cache": hf_cache,
        "/models": longcat_models,
        "/dramabox_outputs": dramabox_outputs,
        "/outputs": longcat_outputs,
    },
)
def run_space_progress_avatar(
    run_id: str = "20260522T043346Z",
    output_name: str = "space_progress_longcat_avatar_15_attempt2",
) -> dict:
    import json
    import os
    import subprocess
    from pathlib import Path

    from huggingface_hub import snapshot_download

    repo = Path(REMOTE_REPO)
    model_dir = Path("/models/LongCat-Video-Avatar-1.5")
    output_dir = Path("/outputs") / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Verifying DramaBox stems ===", flush=True)
    stem_root = Path("/dramabox_outputs/intergalactic-signal-check") / run_id
    person1 = stem_root / "01_mindexpander_signal_open.wav"
    person2 = stem_root / "02_nasa_radio_reply.wav"
    for p in [person1, person2]:
        print(p, p.exists(), p.stat().st_size if p.exists() else "missing", flush=True)
        if not p.exists():
            raise FileNotFoundError(str(p))

    print("=== Ensuring LongCat base + Avatar-1.5 weights ===", flush=True)
    # The official avatar demo loads shared tokenizer/text/VAE components from
    # os.path.join(checkpoint_dir, '..', 'LongCat-Video'), so both sibling dirs
    # must exist in /models.
    base_model_dir = Path("/models/LongCat-Video")
    if not (base_model_dir / "tokenizer").exists():
        snapshot_download(
            "meituan-longcat/LongCat-Video",
            local_dir=str(base_model_dir),
            local_dir_use_symlinks=False,
            allow_patterns=[
                "tokenizer/*",
                "text_encoder/*",
                "vae/*",
                "scheduler/*",
                "*.json",
                "*.md",
                "LICENSE",
            ],
        )
    if not (model_dir / "dit").exists():
        snapshot_download(
            "meituan-longcat/LongCat-Video-Avatar-1.5",
            local_dir=str(model_dir),
            local_dir_use_symlinks=False,
        )
    print("base_model_dir contents:", [p.name for p in base_model_dir.iterdir()][:20], flush=True)
    print("avatar_model_dir contents:", [p.name for p in model_dir.iterdir()][:20], flush=True)

    cond_image = repo / "assets/jimsky/moon_two_astronauts_ref.jpeg"
    if not cond_image.exists():
        raise FileNotFoundError(str(cond_image))

    input_json = output_dir / "space_progress_input.json"
    data = {
        "prompt": (
            "Static camera, two astronauts stand on the moon facing each other, Earthrise in the background, "
            "lunar dust and stars, cinematic realistic lighting. MindExpander reports calm space progress while "
            "a crunchy NASA Apollo mission-control radio voice answers through telemetry static. The scene feels funny, "
            "hopeful, technical, and cosmic, like a space progress checkpoint broadcast from the lunar surface."
        ),
        "cond_image": str(cond_image),
        "cond_audio": {"person1": str(person1), "person2": str(person2)},
        "audio_type": "add",
    }
    input_json.write_text(json.dumps(data, indent=2))

    cmd = [
        "torchrun", "--standalone", "--nproc_per_node=1",
        "run_demo_avatar_multi_audio_to_video.py",
        "--input_json", str(input_json),
        "--output_dir", str(output_dir),
        "--resolution", "480p",
        "--num_segments", "1",
        "--checkpoint_dir", str(model_dir),
        "--model_type", "avatar-v1.5",
        "--use_distill",
    ]
    print("RUN:", " ".join(cmd), flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=str(repo),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env={**os.environ, "PYTHONPATH": str(repo)},
        bufsize=1,
    )
    assert proc.stdout is not None
    tail = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        tail.append(line)
        tail = tail[-200:]
    rc = proc.wait()
    if rc != 0:
        (output_dir / "failure_tail.log").write_text("".join(tail))
        raise subprocess.CalledProcessError(rc, cmd)

    videos = sorted([str(p) for p in output_dir.rglob("*.mp4")])
    wavs = sorted([str(p) for p in output_dir.rglob("*.wav")])
    receipt = {
        "status": "ok" if videos else "no_video_found",
        "videos": videos,
        "wavs": wavs,
        "input_json": str(input_json),
        "output_dir": str(output_dir),
        "cond_image": str(cond_image),
        "dramabox_stems": [str(person1), str(person2)],
    }
    (output_dir / "receipt.json").write_text(json.dumps(receipt, indent=2))
    print("RECEIPT", json.dumps(receipt, indent=2), flush=True)
    return receipt


@app.local_entrypoint()
def main(run_id: str = "20260522T043346Z"):
    print(run_space_progress_avatar.remote(run_id=run_id))
