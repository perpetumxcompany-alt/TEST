"""How fast is Qwen3-TTS on CPU only? Real-time factor for 0.6B vs 1.7B (Base, voice clone), fp32 vs bf16.
RTF here = seconds of audio produced per second of compute (>1 means faster than real time).
Usage: python ci/qwen_cpu_bench.py <ref.wav>
"""
import json, os, platform, subprocess, sys, time

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

ref = sys.argv[1]
cfg = json.load(open("tts/config.json", encoding="utf-8"))
torch.set_num_threads(os.cpu_count())
short = "Борис, презентацию для клиента вы уже отправили?"
cpu = subprocess.run("lscpu | grep -m1 'Model name'", shell=True, capture_output=True, text=True).stdout.strip()
os.makedirs("out/bench", exist_ok=True)
results = {"cpu": cpu, "cores": os.cpu_count(), "torch": torch.__version__, "machine": platform.machine(), "runs": []}


def run(model_id, dtype, label, full):
    t0 = time.time()
    m = Qwen3TTSModel.from_pretrained(model_id, device_map="cpu", dtype=dtype)
    load_s = time.time() - t0
    prompt = m.create_voice_clone_prompt(ref_audio=ref, ref_text=cfg["ref_text"], x_vector_only_mode=False)
    m.generate_voice_clone(text=short, language="Russian", voice_clone_prompt=prompt)  # warm-up
    texts = [("short", short)] + ([("full", cfg["text"])] if full else [])
    for kind, text in texts:
        t0 = time.time()
        w, sr = m.generate_voice_clone(text=text, language="Russian", voice_clone_prompt=prompt)
        gen = time.time() - t0
        dur = len(w[0]) / sr
        r = dict(model=label, dtype=str(dtype).replace("torch.", ""), text=kind, audio_s=round(dur, 2),
                 compute_s=round(gen, 2), rtf=round(dur / gen, 3), load_s=round(load_s, 1))
        results["runs"].append(r)
        print(r, flush=True)
        if kind == "full" and dtype == torch.float32:
            sf.write(f"out/bench/qwen3-{label}-cpu-clone.wav", w[0], sr)
        json.dump(results, open("out/bench/cpu-bench.json", "w"), ensure_ascii=False, indent=1)
    del m


run("Qwen/Qwen3-TTS-12Hz-0.6B-Base", torch.float32, "0.6B", full=True)
run("Qwen/Qwen3-TTS-12Hz-0.6B-Base", torch.bfloat16, "0.6B", full=False)
run("Qwen/Qwen3-TTS-12Hz-1.7B-Base", torch.float32, "1.7B", full=True)
os._exit(0)
