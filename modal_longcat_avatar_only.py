"""
Lightweight LongCat-Video-Avatar-1.5 runner.
Takes pre-generated DramaBox audio stems and produces moon scene video.
"""

import modal
from pathlib import Path

hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
longcat_outputs = modal.Volume.from_name("longcat-avatar-outputs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libsndfile1", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.8.0+cu128",
        "torchaudio==2.8.0+cu128",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "diffusers>=0.33.0",
        "transformers==4.57.1",
        "accelerate",
        "huggingface_hub[hf_transfer]",
        "librosa",
        "soundfile",
        "einops",
        "pillow",
        "numpy",
    )
    .run_commands("pip install flash-attn==2.7.4.post1 --no-build-isolation")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App("longcat-avatar-moon")

@app.function(
    image=image,
    gpu="H100",
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/outputs": longcat_outputs,
    },
    timeout=3600,
)
def generate_avatar_from_audio(
    audio_paths: list[str],
    prompt: str,
    output_name: str = "moon_avatar",
):
    import json
    import subprocess
    from pathlib import Path

    output_dir = Path("/outputs") / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    input_json = {
        "prompt": prompt,
        "negative_prompt": "blurry, low quality, deformed, text, watermark",
        "cond_audio": {f"person{i+1}": p for i, p in enumerate(audio_paths)},
        "model_type": "avatar-v1.5",
        "use_distill": True,
        "resolution": "720p",
    }

    json_path = output_dir / "input.json"
    json_path.write_text(json.dumps(input_json, indent=2))

    cmd = [
        "python", "run_demo_avatar_multi_audio_to_video.py",
        "--input_json", str(json_path),
        "--checkpoint_dir", "/models/longcat-avatar-1.5",
        "--model_type", "avatar-v1.5",
        "--use_distill",
        "--resolution", "720p",
        "--num_inference_steps", "8",
        "--output_dir", str(output_dir),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, cwd="/opt/data/workspace/github-forks/LongCat-Video-Avatar")

    print(result.stdout[-2000:] if result.stdout else "")
    if result.returncode != 0:
        print(result.stderr[-2000:] if result.stderr else "")
        raise RuntimeError("LongCat failed")

    videos = list(output_dir.glob("*.mp4"))
    return {"video": str(videos[0]) if videos else None, "output_dir": str(output_dir)}


@app.local_entrypoint()
def main():
    # Example using previously generated DramaBox stems
    audio = [
        "/outputs/intergalactic-signal-check/20260522T043346Z/01_mindexpander_signal_open.wav",
        "/outputs/intergalactic-signal-check/20260522T043346Z/02_nasa_radio_reply.wav",
    ]
    prompt = "MindExpander and NASA radio operator talking on the moon at night, Earthrise in background, cinematic, lunar dust, high detail"
    result = generate_avatar_from_audio.remote(audio_paths=audio, prompt=prompt, output_name="moon_test_v1")
    print(result)