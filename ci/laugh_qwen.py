"""Can Qwen3-TTS laugh? Calm Serena with a laugh written into the text.
1.7B-CustomVoice: instruction asks for a soft natural laugh at "ха-ха" (instructions work only in 1.7B).
0.6B-Base: clone of the calm Serena reference, laugh only from the text (no instruction support).
Usage: python ci/laugh_qwen.py custom | clone <ref.wav>
"""
import json, os, sys, time

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

mode = sys.argv[1]
text = "Борис, вы опять поставили на утро двенадцать задач? Ха-ха-ха. Ну хорошо, давайте хотя бы с первой начнём."
ref_text = "Здравствуйте, Борис. Меня зовут Лора, я ваш личный ассистент. Расскажите, чем я могу помочь сегодня? Я всё запомню и ничего не упущу."
out = "out/laugh"
os.makedirs(out, exist_ok=True)
torch.set_num_threads(os.cpu_count())


def save(name, wav, sr, t0, **kw):
    sf.write(f"{out}/{name}.wav", wav, sr)
    meta = dict(name=name, engine="Qwen3-TTS", gen_s=round(time.time() - t0, 1), dur_s=round(len(wav) / sr, 1),
                sr=sr, text=text, **kw)
    json.dump([meta], open(f"{out}/meta-{name}.json", "w"), ensure_ascii=False, indent=1)
    print(meta, flush=True)


if mode == "custom":
    m = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", device_map="cpu", dtype=torch.float32)
    instruct = ("Speak Russian calmly and warmly, like a composed personal assistant gently teasing a friend. "
                "At 'ха-ха-ха' give a soft, short, genuine laugh, then continue calmly with a smile in the voice.")
    t0 = time.time()
    w, sr = m.generate_custom_voice(text=text, language="Russian", speaker="Serena", instruct=instruct)
    save("laugh-qwen17-serena", w[0], sr, t0, model="1.7B-CustomVoice", instruct=instruct)
else:
    m = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-Base", device_map="cpu", dtype=torch.float32)
    prompt = m.create_voice_clone_prompt(ref_audio=sys.argv[2], ref_text=ref_text, x_vector_only_mode=False)
    t0 = time.time()
    w, sr = m.generate_voice_clone(text=text, language="Russian", voice_clone_prompt=prompt)
    save("laugh-qwen06-clone", w[0], sr, t0, model="0.6B-Base", reference="serena_calm_ref")
os._exit(0)
