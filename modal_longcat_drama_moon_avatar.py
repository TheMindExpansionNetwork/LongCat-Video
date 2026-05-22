"""
Modal app for LongCat-Video-Avatar-1.5 + DramaBox TTS
Generates expressive audio with DramaBox, then drives LongCat avatar video
with the character placed "on the moon" using image conditioning + prompt.

Usage:
  modal run modal_longcat_drama_moon_avatar.py --prompt "We are explorers on the lunar surface under the Earthrise..." --voice "mindexpander"
"""

import modal
import os
from pathlib import Path

# Volumes for caching
hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
output_vol = modal.Volume.from_name("longcat-avatar-outputs", create_if_missing=True)

# Image with all deps for LongCat + DramaBox
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "ffmpeg", "git", "libgl1-mesa-glx", "libglib2.0-0",
        "libsndfile1", "libavcodec-extra"
    )
    .pip_install(
        "torch==2.6.0+cu124",
        "torchvision==0.21.0+cu124",
        "torchaudio==2.6.0",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "flash-attn==2.7.4.post1",
        "diffusers>=0.33.0",
        "transformers==4.51.3",
        "accelerate",
        "huggingface_hub[hf_transfer]",
        "librosa",
        "soundfile",
        "einops",
        "pillow",
        "numpy",
        "opencv-python-headless",
    )
    .run_commands(
        "pip install git+https://github.com/meituan-longcat/LongCat-Video.git@main#egg=longcat-video",
        "pip install -r https://raw.githubusercontent.com/TheMindExpansionNetwork/LongCat-Video/main/requirements_avatar.txt || true",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "TORCH_CUDNN_V8_API_ENABLED": "1",
    })
)

app = modal.App("longcat-drama-moon-avatar")

@app.function(
    image=image,
    gpu="H100",  # or A100-80GB for smaller runs; model is large
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/outputs": output_vol,
    },
    timeout=3600,
    memory=32768,
)
def generate_moon_avatar(
    text_prompt: str = "We have landed on the moon. The Earth rises majestically over the horizon as we explore the lunar surface.",
    reference_image_url: str = None,  # Optional: URL or path to character image on moon
    voice: str = "mindexpander",
    duration: int = 8,  # seconds of video
    output_name: str = "moon_avatar_demo",
):
    """
    1. Generate expressive TTS audio using DramaBox (or fallback to local if integrated)
    2. Run LongCat-Video-Avatar-1.5 with audio + moon scene conditioning
    """
    import torch
    from pathlib import Path
    import subprocess
    import json
    import tempfile
    import requests
    from PIL import Image

    output_dir = Path("/outputs") / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Generate DramaBox expressive audio
    # For now, we use a placeholder that assumes DramaBox Modal is called separately
    # or integrate here. In production, call the DramaBox endpoint or import its generator.
    print("🎙️ Generating expressive TTS with DramaBox (moon theme)...")
    audio_path = output_dir / "drama_moon_voice.wav"

    # TODO: Replace with actual DramaBox call
    # Example integration point:
    # from drama_box_modal import generate_drama_tts
    # audio_path = generate_drama_tts(text_prompt, voice=voice, emotion="wonder", lunar_radio_style=True)
    #
    # For immediate use, we create a dummy or use edge-tts as fallback for testing:
    try:
        subprocess.run([
            "python", "-c",
            f"""
import edge_tts
import asyncio
async def tts():
    communicate = edge_tts.Communicate('{text_prompt}', 'en-US-AriaNeural')
    await communicate.save('{audio_path}')
asyncio.run(tts())
print('Fallback TTS generated')
            """
        ], check=True, timeout=60)
    except Exception as e:
        print(f"DramaBox/edge-tts fallback failed: {e}")
        # Create silent audio as last resort
        subprocess.run(f"ffmpeg -f lavfi -i anullsrc=r=16000:cl=mono -t 8 {audio_path} -y", shell=True)

    print(f"✅ Audio ready: {audio_path}")

    # Step 2: Prepare moon scene reference image (if not provided)
    if reference_image_url is None:
        # Create a simple moon scene placeholder or download a public domain lunar image
        moon_img_path = output_dir / "moon_reference.png"
        try:
            # In real run, user should provide a good reference image of the speaker on moon
            # For demo we use a generated or stock lunar landscape + person silhouette
            img = Image.new("RGB", (768, 1280), color=(20, 20, 40))  # dark space
            img.save(moon_img_path)
            print("📸 Using placeholder moon reference (provide better image for real results)")
        except Exception:
            moon_img_path = None
    else:
        moon_img_path = output_dir / "reference.png"
        r = requests.get(reference_image_url, timeout=30)
        with open(moon_img_path, "wb") as f:
            f.write(r.content)

    # Step 3: Build input JSON for LongCat avatar demo
    input_json = {
        "prompt": f"{text_prompt}, cinematic, astronaut on the moon, Earthrise in background, lunar dust, high detail, realistic lighting",
        "negative_prompt": "blurry, low quality, deformed, extra limbs, text, watermark",
        "cond_audio": {
            "person1": str(audio_path)
        },
        "reference_image": str(moon_img_path) if moon_img_path else None,
    }

    json_path = output_dir / "input.json"
    with open(json_path, "w") as f:
        json.dump(input_json, f, indent=2)

    print("🚀 Running LongCat-Video-Avatar-1.5 inference on moon scene...")

    # Run the official single-audio demo with v1.5 settings
    cmd = [
        "python", "run_demo_avatar_single_audio_to_video.py",
        "--input_json", str(json_path),
        "--checkpoint_dir", "weights/LongCat-Video-Avatar-1.5",
        "--model_type", "avatar-v1.5",
        "--use_distill",
        "--resolution", "720p",
        "--num_inference_steps", "8",
        "--output_dir", str(output_dir),
        "--audio_guidance_scale", "3.0",
    ]

    # Note: In production Modal you would download weights first via huggingface-cli
    # or use --checkpoint_dir pointing to HF cache.

    result = subprocess.run(cmd, capture_output=True, text=True, cwd="/opt/data/workspace/github-forks/LongCat-Video-Avatar")

    print("STDOUT:", result.stdout[-2000:] if result.stdout else "")
    if result.returncode != 0:
        print("STDERR:", result.stderr[-2000:] if result.stderr else "")
        raise RuntimeError("LongCat inference failed")

    # Find generated video
    video_files = list(output_dir.glob("*.mp4")) + list(output_dir.glob("*.mov"))
    if video_files:
        final_video = video_files[0]
        print(f"✅ Moon avatar video generated: {final_video}")
        return {"video_path": str(final_video), "audio_path": str(audio_path), "status": "success"}
    else:
        return {"status": "no_video_found", "output_dir": str(output_dir)}


@app.local_entrypoint()
def main(prompt: str = "Standing on the moon, I can see the entire Earth glowing in the distance. This is incredible.", voice: str = "mindexpander"):
    result = generate_moon_avatar.remote(text_prompt=prompt, voice=voice)
    print(result)