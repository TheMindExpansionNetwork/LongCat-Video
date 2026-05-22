"""LongCat-Video-Avatar-1.5 drama/emotion skit on Modal.

Uses clean validated local TTS stems instead of the broken clipped DramaBox stems.
"""
from __future__ import annotations

import modal

REPO_ROOT = "/opt/data/workspace/github-forks/LongCat-Video-Avatar"
REMOTE_REPO = "/workspace/LongCat-Video"

app = modal.App("longcat-avatar-15-drama-skit")

hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
longcat_models = modal.Volume.from_name("longcat-avatar-models", create_if_missing=True)
longcat_outputs = modal.Volume.from_name("longcat-avatar-outputs", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "ffmpeg", "libsndfile1", "libgl1", "libglib2.0-0", "ninja-build", "curl")
    .run_commands(
        "python -m pip install --upgrade pip setuptools wheel packaging psutil ninja",
        "pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124",
        "pip install numpy==1.26.4 transformers==4.41.0 diffusers==0.35.1 einops==0.8.0 ftfy==6.2.0 psutil==6.0.0 av==12.0.0 opencv-python-headless==4.9.0.80 imageio==2.37.0 imageio-ffmpeg==0.6.0 accelerate",
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
        "/outputs": longcat_outputs,
    },
)
def run_drama_skit(output_name: str = "drama_emotion_skit_avatar_15") -> dict:
    import json
    import os
    import subprocess
    import wave
    from pathlib import Path

    from huggingface_hub import snapshot_download

    repo = Path(REMOTE_REPO)
    model_dir = Path("/models/LongCat-Video-Avatar-1.5")
    base_model_dir = Path("/models/LongCat-Video")
    output_dir = Path("/outputs") / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Ensuring model layout ===", flush=True)
    if not (base_model_dir / "tokenizer").exists() or not (base_model_dir / "vae").exists():
        snapshot_download(
            "meituan-longcat/LongCat-Video",
            local_dir=str(base_model_dir),
            local_dir_use_symlinks=False,
            allow_patterns=["tokenizer/*", "text_encoder/*", "vae/*", "scheduler/*", "*.json", "*.md", "LICENSE"],
        )
    if not (model_dir / "base_model").exists():
        snapshot_download(
            "meituan-longcat/LongCat-Video-Avatar-1.5",
            local_dir=str(model_dir),
            local_dir_use_symlinks=False,
        )

    person1 = repo / "assets/jimsky/drama_skit/person1_drama.wav"
    person2 = repo / "assets/jimsky/drama_skit/person2_nasa_reply.wav"
    cond_image = repo / "assets/jimsky/moon_two_astronauts_ref.jpeg"
    for p in [person1, person2, cond_image]:
        if not p.exists():
            raise FileNotFoundError(str(p))
    for wav_path in [person1, person2]:
        with wave.open(str(wav_path), "rb") as w:
            print("AUDIO", wav_path, {"sr": w.getframerate(), "ch": w.getnchannels(), "dur": round(w.getnframes()/w.getframerate(), 3)}, flush=True)

    input_json = output_dir / "input.json"
    data = {
        "prompt": (
            "Static camera, two expressive astronauts on the moon facing each other, Earthrise behind them, cinematic lunar dust, "
            "one astronaut is wildly dramatic and emotional, asking if the scene has enough drama; the other answers like a NASA radio operator, dry and funny. "
            "The performance is theatrical, goofy, emotional, and cosmic."
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
        "--num_segments", "5",
        "--checkpoint_dir", str(model_dir),
        "--model_type", "avatar-v1.5",
        "--use_distill",
    ]
    print("RUN:", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, cwd=str(repo), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env={**os.environ, "PYTHONPATH": str(repo)}, bufsize=1)
    tail = []
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        tail.append(line)
        tail = tail[-240:]
    rc = proc.wait()
    if rc != 0:
        (output_dir / "failure_tail.log").write_text("".join(tail))
        raise subprocess.CalledProcessError(rc, cmd)

    videos = sorted([str(p) for p in output_dir.rglob("*.mp4")])
    receipt = {"status": "ok" if videos else "no_video_found", "videos": videos, "input_json": str(input_json), "output_dir": str(output_dir)}
    (output_dir / "receipt.json").write_text(json.dumps(receipt, indent=2))
    print("RECEIPT", json.dumps(receipt, indent=2), flush=True)
    return receipt


@app.local_entrypoint()
def main():
    print(run_drama_skit.remote())
