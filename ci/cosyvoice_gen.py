"""Fun-CosyVoice3-0.5B on CPU, zero-shot cloning from one reference clip.
Saves with soundfile: torchaudio 2.3.1 segfaults with the runner's sox 14.7 when saving.
Usage: python ci/cosyvoice_gen.py <name> <ref.wav>
"""
import json, os, sys, time

sys.path.append("CosyVoice")
sys.path.append("CosyVoice/third_party/Matcha-TTS")
import soundfile as sf
import torch
from cosyvoice.cli.cosyvoice import AutoModel

name, ref = sys.argv[1:3]
cfg = json.load(open("tts/config.json", encoding="utf-8"))
out = "out/cosyvoice"
os.makedirs(out, exist_ok=True)
torch.set_num_threads(os.cpu_count())
model = AutoModel(model_dir="pretrained_models/Fun-CosyVoice3-0.5B")
prompt_text = "You are a helpful assistant.<|endofprompt|>" + cfg["ref_text"]
t0 = time.time()
chunks = [j["tts_speech"] for j in model.inference_zero_shot(cfg["text"], prompt_text, ref, stream=False)]
wav = torch.cat(chunks, dim=1).squeeze(0).cpu().numpy()
sf.write(f"{out}/{name}.wav", wav, model.sample_rate)
meta = [dict(name=name, engine="CosyVoice 3", model="Fun-CosyVoice3-0.5B-2512", reference=ref,
             gen_s=round(time.time() - t0, 1), dur_s=round(len(wav) / model.sample_rate, 1),
             sr=model.sample_rate, cpus=os.cpu_count())]
json.dump(meta, open(f"{out}/meta-{name}.json", "w"), ensure_ascii=False, indent=1)
print(meta[0], flush=True)
os._exit(0)  # skip interpreter teardown, where the old torch build has also crashed
