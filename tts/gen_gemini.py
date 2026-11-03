"""Gemini TTS through the Gemini API (needs the GEMINI_API_KEY secret).
Picks the TTS models the key can see, prefers the newest Flash TTS, and reads the text with several voices.
"""
import base64, json, os, sys, time, urllib.error, urllib.request, wave

key = os.environ.get("GEMINI_API_KEY", "").strip()
if not key:
    # Fail on purpose: once the secret is added, "Re-run failed jobs" runs only Gemini and the collector.
    print("::error::GEMINI_API_KEY secret is not set. Add it in Settings > Secrets and variables > Actions.")
    sys.exit(1)

cfg = json.load(open("tts/config.json", encoding="utf-8"))
base = "https://generativelanguage.googleapis.com/v1beta"
out = "out/gemini"
os.makedirs(out, exist_ok=True)


def call(url, body=None):
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body else None,
                                 headers={"x-goog-api-key": key, "Content-Type": "application/json"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:500]
            if e.code in (429, 500, 503) and attempt < 5:
                print("retry", e.code, msg[:160], flush=True)
                time.sleep(20 * (attempt + 1))
                continue
            raise SystemExit(f"Gemini API error {e.code}: {msg}")


models = [m["name"].split("/", 1)[1] for m in call(f"{base}/models?pageSize=200").get("models", [])
          if "tts" in m["name"] and "generateContent" in m.get("supportedGenerationMethods", [])]
print("TTS models visible to this key:", models, flush=True)
flash = sorted([m for m in models if "flash" in m], reverse=True)
pro = sorted([m for m in models if "pro" in m], reverse=True)
plan = []
if flash:
    plan += [(flash[0], v) for v in ("Kore", "Aoede", "Leda", "Sulafat", "Despina", "Zephyr")]
if pro:
    plan += [(pro[0], v) for v in ("Kore", "Sulafat")]
if not plan:
    raise SystemExit("No TTS models are available for this key.")

prompt = ("Read the following Russian text aloud as a warm, composed personal assistant: crisp diction, "
          "rich melodic intonation, natural Russian question intonation, bright but not theatrical.\n\n" + cfg["text"])
meta = []
for model, voice in plan:
    t0 = time.time()
    r = call(f"{base}/models/{model}:generateContent", {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseModalities": ["AUDIO"],
                             "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}}}})
    part = r["candidates"][0]["content"]["parts"][0]["inlineData"]
    pcm = base64.b64decode(part["data"])
    rate = int(part.get("mimeType", "rate=24000").split("rate=")[-1].split(";")[0] or 24000)
    name = f"gemini-{model.replace('gemini-', '').replace('-preview', '')}-{voice.lower()}"
    with wave.open(f"{out}/{name}.wav", "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate); w.writeframes(pcm)
    meta.append(dict(name=name, engine="Gemini TTS", model=model, voice=voice, gen_s=round(time.time() - t0, 1),
                     dur_s=round(len(pcm) / 2 / rate, 1), sr=rate))
    print(meta[-1], flush=True)
    json.dump(meta, open(f"{out}/meta.json", "w"), ensure_ascii=False, indent=1)
    time.sleep(7)  # stay under free-tier per-minute limits
