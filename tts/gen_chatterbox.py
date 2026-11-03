"""Chatterbox Multilingual V3 on CPU, language_id="ru".
Generates sentence by sentence (more stable for long text) and joins with short pauses.
Usage: python tts/gen_chatterbox.py <name>=<ref.wav|default> [...]
"""
import json, os, re, sys, time

import torch

_load = torch.load


def _cpu_load(*a, **k):  # checkpoints may be saved from CUDA
    k.setdefault("map_location", "cpu")
    return _load(*a, **k)


torch.load = _cpu_load
import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

cfg = json.load(open("tts/config.json", encoding="utf-8"))
out = "out/chatterbox"
os.makedirs(out, exist_ok=True)
torch.set_num_threads(os.cpu_count())
model = ChatterboxMultilingualTTS.from_pretrained(device="cpu", t3_model="v3")
sentences = [s for s in re.split(r"(?<=[.!?])\s+", cfg["text"]) if s.strip()]
pause = torch.zeros(1, int(0.22 * model.sr))
meta = []
for arg in sys.argv[1:]:
    name, ref = arg.split("=", 1)
    prompt = None if ref == "default" else ref
    if prompt and not os.path.isfile(prompt):
        print("skip, no reference:", prompt, flush=True)
        continue
    t0, parts = time.time(), []
    for s in sentences:
        wav = model.generate(s, language_id="ru", audio_prompt_path=prompt, exaggeration=0.6, cfg_weight=0.5)
        parts += [wav, pause]
    wav = torch.cat(parts[:-1], dim=1)
    ta.save(f"{out}/{name}.wav", wav, model.sr)
    meta.append(dict(name=name, engine="Chatterbox", model="Multilingual V3", reference=ref,
                     exaggeration=0.6, cfg_weight=0.5, gen_s=round(time.time() - t0, 1),
                     dur_s=round(wav.shape[1] / model.sr, 1), sr=model.sr, cpus=os.cpu_count()))
    print(meta[-1], flush=True)
    json.dump(meta, open(f"{out}/meta.json", "w"), ensure_ascii=False, indent=1)
