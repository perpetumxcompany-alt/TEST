"""Calm Laura voices with Qwen3-TTS on CPU.
design: four VoiceDesign descriptions of a calm, medium-high female voice -> short reference clip each,
        then the Base model clones every reference on the test text ("Voice Design then Clone").
serena: the built-in Serena voice with a calm instruction.
Usage: python ci/qwen_calm.py design|serena
"""
import gc, json, os, sys, time

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

mode = sys.argv[1]
cfg = json.load(open("tts/config.json", encoding="utf-8"))
text = cfg["text"]
ref_text = "Здравствуйте, Борис. Меня зовут Лора, я ваш личный ассистент. Расскажите, чем я могу помочь сегодня? Я всё запомню и ничего не упущу."
base = ("A native Russian speaker with clear, standard literary pronunciation. Studio-quality close-microphone "
        "recording, no background noise. ")
designs = {
    "calm1": base + "A calm, soft-spoken woman in her early thirties. Medium-high feminine pitch, but relaxed and "
             "unhurried, gentle low-energy delivery. Warm, slightly velvety timbre with a light breathiness. Smooth, "
             "even intonation, soft natural rises on questions, settled falling endings. Never cheerful or excited.",
    "calm2": base + "A composed, confident woman of about thirty-five who speaks like a late-night radio host: "
             "low-key, measured pace, soft falling cadences, warm velvety mid-high female timbre, quiet assurance. "
             "Questions rise gently, without any sing-song energy.",
    "calm3": base + "A gentle, serene young woman with a quiet, tender voice: medium-high pitch, slightly husky and "
             "airy, slow and thoughtful pace, minimal pitch jumps, calm and reassuring, intimate but clear.",
    "calm4": base + "An elegant, calm professional female voice, a little lower than average female pitch but clearly "
             "feminine, with warm chest resonance. Steady, soothing, unhurried delivery; restrained, natural "
             "intonation with clear but soft question rises.",
}
out = "out/qwen-calm"
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
    for key, desc in designs.items():
        t0 = time.time()
        w, sr = m.generate_voice_design(text=ref_text, language="Russian", instruct=desc)
        save(f"{key}_ref", w[0], sr, t0, model="1.7B-VoiceDesign", role="reference", description=desc)
    del m
    gc.collect()
    m = load("Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    for key in designs:
        prompt = m.create_voice_clone_prompt(ref_audio=f"{out}/{key}_ref.wav", ref_text=ref_text,
                                             x_vector_only_mode=False)
        t0 = time.time()
        w, sr = m.generate_voice_clone(text=text, language="Russian", voice_clone_prompt=prompt)
        save(f"qwen3-{key}-clone", w[0], sr, t0, model="1.7B-Base", reference=f"{key}_ref")
else:
    m = load("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    instruct = ("Speak Russian calmly and softly, unhurried and low-energy, like a composed personal assistant. "
                "Gentle, natural question rises; settled falling endings; no excitement.")
    t0 = time.time()
    w, sr = m.generate_custom_voice(text=text, language="Russian", speaker="Serena", instruct=instruct)
    save("qwen3-serena-calm", w[0], sr, t0, model="1.7B-CustomVoice", speaker="Serena", instruct=instruct)
os._exit(0)
