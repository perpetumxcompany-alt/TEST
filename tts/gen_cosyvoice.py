"""Fun-CosyVoice3-0.5B on CPU, zero-shot cloning from a reference clip.
Usage: python tts/gen_cosyvoice.py <name>=<ref.wav> [<name>=<ref.wav> ...]
Run from the workspace root with the CosyVoice repo checked out in ./CosyVoice.
"""
import json, os, sys, time

sys.path.append("CosyVoice")
sys.path.append("CosyVoice/third_party/Matcha-TTS")
import torch
import torchaudio
from cosyvoice.cli.cosyvoice import AutoModel

cfg = json.load(open("tts/config.json", encoding="utf-8"))
out = "out/cosyvoice"
os.makedirs(out, exist_ok=True)
torch.set_num_threads(os.cpu_count())
model = AutoModel(model_dir="pretrained_models/Fun-CosyVoice3-0.5B")
prompt_text = "You are a helpful assistant.<|endofprompt|>" + cfg["ref_text"]
meta = []
for arg in sys.argv[1:]:
    name, ref = arg.split("=", 1)
    if not os.path.isfile(ref):
        print("skip, no reference:", ref, flush=True)
        continue
    t0 = time.time()
    chunks = [j["tts_speech"] for j in model.inference_zero_shot(cfg["text"], prompt_text, ref, stream=False)]
    wav = torch.cat(chunks, dim=1)
    torchaudio.save(f"{out}/{name}.wav", wav, model.sample_rate)
    meta.append(dict(name=name, engine="CosyVoice 3", model="Fun-CosyVoice3-0.5B-2512", reference=ref,
                     gen_s=round(time.time() - t0, 1), dur_s=round(wav.shape[1] / model.sample_rate, 1),
                     sr=model.sample_rate, cpus=os.cpu_count()))
    print(meta[-1], flush=True)
    json.dump(meta, open(f"{out}/meta.json", "w"), ensure_ascii=False, indent=1)
