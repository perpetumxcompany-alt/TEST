"""Calm Serena cloned by the light 0.6B model (CPU).
1. 1.7B-CustomVoice speaks a short reference in calm Serena (exact transcript known).
2. 0.6B-Base clones it: two fp32 takes of the test text (stability), one bf16 take (speed), a morning brief.
3. 1.7B-Base clones the same reference once, for comparison.
Every result records the real-time factor (seconds of audio per second of compute).
"""
import gc, json, os, time

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

cfg = json.load(open("tts/config.json", encoding="utf-8"))
ref_text = "Здравствуйте, Борис. Меня зовут Лора, я ваш личный ассистент. Расскажите, чем я могу помочь сегодня? Я всё запомню и ничего не упущу."
brief = ("Доброе утро, Борис. Сегодня среда, четвёртое ноября. Главный приоритет — стратегия для клиента: "
         "черновик нужно закончить до обеда. В два часа созвон по сайту, а вечером остаётся свободное окно для чтения. "
         "Вчера вы закрыли четыре задачи из пяти, незакрытую я перенесла на сегодня. Хорошего дня.")
instruct = ("Speak Russian calmly and softly, unhurried and low-energy, like a composed personal assistant. "
            "Gentle, natural question rises; settled falling endings; no excitement.")
out = "out/serena-clone"
os.makedirs(out, exist_ok=True)
torch.set_num_threads(os.cpu_count())
meta = []


def save(name, wav, sr, t0, **kw):
    gen = time.time() - t0
    sf.write(f"{out}/{name}.wav", wav, sr)
    meta.append(dict(name=name, engine="Qwen3-TTS", gen_s=round(gen, 1), dur_s=round(len(wav) / sr, 1),
                     rtf=round(len(wav) / sr / gen, 3), sr=sr, cpus=os.cpu_count(), **kw))
    print(meta[-1], flush=True)
    json.dump(meta, open(f"{out}/meta.json", "w"), ensure_ascii=False, indent=1)


def load(name, dtype=torch.float32):
    return Qwen3TTSModel.from_pretrained(name, device_map="cpu", dtype=dtype)


m = load("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
t0 = time.time()
w, sr = m.generate_custom_voice(text=ref_text, language="Russian", speaker="Serena", instruct=instruct)
save("serena_calm_ref", w[0], sr, t0, model="1.7B-CustomVoice", role="reference")
ref = f"{out}/serena_calm_ref.wav"
del m
gc.collect()


def clone_runs(model_id, label, dtype, jobs):
    m = load(model_id, dtype)
    prompt = m.create_voice_clone_prompt(ref_audio=ref, ref_text=ref_text, x_vector_only_mode=False)
    m.generate_voice_clone(text="Добрый вечер, Борис.", language="Russian", voice_clone_prompt=prompt)  # warm-up
    for name, text in jobs:
        t0 = time.time()
        w, sr = m.generate_voice_clone(text=text, language="Russian", voice_clone_prompt=prompt)
        save(name, w[0], sr, t0, model=label, dtype=str(dtype).replace("torch.", ""), reference="serena_calm_ref")
    del m
    gc.collect()


clone_runs("Qwen/Qwen3-TTS-12Hz-0.6B-Base", "0.6B-Base", torch.float32,
           [("serena06-take1", cfg["text"]), ("serena06-take2", cfg["text"])])
clone_runs("Qwen/Qwen3-TTS-12Hz-0.6B-Base", "0.6B-Base", torch.bfloat16,
           [("serena06-bf16", cfg["text"]), ("serena06-brief", brief)])
clone_runs("Qwen/Qwen3-TTS-12Hz-1.7B-Base", "1.7B-Base", torch.float32, [("serena17-clone", cfg["text"])])
os._exit(0)
