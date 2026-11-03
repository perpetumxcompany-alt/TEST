"""Qwen3-TTS on CPU.
design: VoiceDesign builds the "Laura" voice from a text description (reference clip + two takes),
        then the Base model clones that reference for a consistent voice ("Voice Design then Clone").
custom: built-in CustomVoice speakers with an instruction.
Usage: python tts/gen_qwen.py design|custom
"""
import gc, json, os, sys, time

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

mode = sys.argv[1]
cfg = json.load(open("tts/config.json", encoding="utf-8"))
out = "out/qwen"
os.makedirs(out, exist_ok=True)
torch.set_num_threads(os.cpu_count())
meta = []


def save(name, wav, sr, t0, **kw):
    sf.write(f"{out}/{name}.wav", wav, sr)
    meta.append(dict(name=name, engine="Qwen3-TTS", gen_s=round(time.time() - t0, 1),
                     dur_s=round(len(wav) / sr, 1), sr=sr, cpus=os.cpu_count(), **kw))
    print(meta[-1], flush=True)
    json.dump(meta, open(f"{out}/meta-{mode}.json", "w"), ensure_ascii=False, indent=1)


def load(name):
    return Qwen3TTSModel.from_pretrained(name, device_map="cpu", dtype=torch.float32)


if mode == "design":
    m = load("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
    t0 = time.time()
    w, sr = m.generate_voice_design(text=cfg["ref_text"], language="Russian", instruct=cfg["voice_design"])
    save("laura_ref", w[0], sr, t0, model="1.7B-VoiceDesign", role="reference")
    for take in (1, 2):
        t0 = time.time()
        w, sr = m.generate_voice_design(text=cfg["text"], language="Russian", instruct=cfg["voice_design"])
        save(f"qwen3-voicedesign-take{take}", w[0], sr, t0, model="1.7B-VoiceDesign")
    del m
    gc.collect()

    m = load("Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    prompt = m.create_voice_clone_prompt(ref_audio=f"{out}/laura_ref.wav", ref_text=cfg["ref_text"],
                                         x_vector_only_mode=False)
    for take in (1, 2):
        t0 = time.time()
        w, sr = m.generate_voice_clone(text=cfg["text"], language="Russian", voice_clone_prompt=prompt)
        save(f"qwen3-design-clone-take{take}", w[0], sr, t0, model="1.7B-Base")
else:
    m = load("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    instruct = "Speak Russian warmly and clearly, like a composed personal assistant, with lively natural question intonation."
    for spk in ("Serena", "Vivian"):
        t0 = time.time()
        w, sr = m.generate_custom_voice(text=cfg["text"], language="Russian", speaker=spk, instruct=instruct)
        save(f"qwen3-custom-{spk.lower()}", w[0], sr, t0, model="1.7B-CustomVoice", speaker=spk)
