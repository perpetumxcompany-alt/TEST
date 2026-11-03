"""Can CosyVoice 3 laugh in Russian? Its tokenizer has [laughter] and <laughter>...</laughter> tags.
Clones the calm Serena reference. Usage: python ci/laugh_cosyvoice.py <ref.wav>
"""
import json, os, sys, time

sys.path.append("CosyVoice")
sys.path.append("CosyVoice/third_party/Matcha-TTS")
import soundfile as sf
import torch
from cosyvoice.cli.cosyvoice import AutoModel

ref = sys.argv[1]
ref_text = "Здравствуйте, Борис. Меня зовут Лора, я ваш личный ассистент. Расскажите, чем я могу помочь сегодня? Я всё запомню и ничего не упущу."
variants = {
    "laugh-cosy-tag": "Борис, вы опять поставили на утро двенадцать задач? [laughter] Ну хорошо, давайте хотя бы с первой начнём.",
    "laugh-cosy-span": "Борис, <laughter>вы опять поставили на утро двенадцать задач?</laughter> Ну хорошо, давайте хотя бы с первой начнём.",
}
out = "out/laugh"
os.makedirs(out, exist_ok=True)
torch.set_num_threads(os.cpu_count())
model = AutoModel(model_dir="pretrained_models/Fun-CosyVoice3-0.5B")
for name, text in variants.items():
    t0 = time.time()
    chunks = [j["tts_speech"] for j in model.inference_zero_shot(
        text, "You are a helpful assistant.<|endofprompt|>" + ref_text, ref, stream=False)]
    wav = torch.cat(chunks, dim=1).squeeze(0).cpu().numpy()
    sf.write(f"{out}/{name}.wav", wav, model.sample_rate)
    meta = dict(name=name, engine="CosyVoice 3", model="Fun-CosyVoice3-0.5B-2512", reference="serena_calm_ref",
                gen_s=round(time.time() - t0, 1), dur_s=round(len(wav) / model.sample_rate, 1),
                sr=model.sample_rate, text=text)
    json.dump([meta], open(f"{out}/meta-{name}.json", "w"), ensure_ascii=False, indent=1)
    print(meta, flush=True)
os._exit(0)
