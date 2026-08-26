# Training the "Adrien" wake word

## Why this step exists

openWakeWord ships pretrained models for a fixed set of phrases — "alexa",
"hey jarvis", "hey mycroft", "hey rhasspy", "timer", "weather". **"Adrien" is
not one of them, and no amount of configuration will make it one.** A wake word
is a trained model, not a string.

So until you train one, Adrien falls back to a pretrained model (`hey_jarvis`
by default) and says so at startup and in `adrien doctor`. Everything else
works; you just say "hey Jarvis" instead of "Adrien". That fallback exists so
the first run is a working assistant rather than a training session.

## The good news

You do not need to record yourself saying "Adrien" hundreds of times.
openWakeWord's training pipeline synthesises its own data: it generates
thousands of TTS utterances of your phrase across many voices and accents,
mixes them with noise and room impulse responses, and trains against a large
corpus of negative audio. It runs free on a Colab GPU in roughly 30–60 minutes.

## Steps

1. Open openWakeWord's automatic model training notebook:
   <https://github.com/dscripka/openWakeWord> → *Training new models* →
   **automatic_model_training.ipynb** (there is a "Open in Colab" badge).

2. Set the target phrase. Include a few spellings so the model learns the sound
   rather than one pronunciation:

   ```python
   target_word = "adrien"
   # Pronunciation variants matter more than exact spelling - the TTS engine
   # reads these phonetically, and "Adrien" is said several ways.
   custom_phrases = ["adrien", "ay-drien", "adrian", "hey adrien"]
   ```

3. Set the runtime to a GPU (Runtime → Change runtime type → T4) and run every
   cell. Generation is the slow part; training itself is minutes.

4. Download the resulting `adrien.onnx`.

5. Drop it into this repo:

   ```bash
   mv ~/Downloads/adrien.onnx models/adrien.onnx
   ```

6. Restart Adrien. It picks the file up automatically — `models/adrien.onnx` is
   already the configured path in `config/settings.json`.

   ```bash
   launchctl kickstart -k gui/$(id -u)/com.raidnxt.adrien
   python -m adrien doctor        # should now show the custom model
   ```

## Tuning the threshold

`config/settings.json`:

```json
"wake_word": {
  "model_path": "models/adrien.onnx",
  "threshold": 0.55,
  "refractory_seconds": 1.5
}
```

- **Triggering on the TV, or on your own conversations?** Raise `threshold`
  towards 0.7.
- **Having to say it twice?** Lower it towards 0.4.
- **Firing several times per "Adrien"?** Raise `refractory_seconds`. One
  utterance spans several 80 ms windows, and this is what stops each of them
  counting separately.

Watch it live while you tune:

```bash
python -m adrien --log-level DEBUG run
```

Every detection logs its score, so you can see how close the near-misses were
rather than guessing.

## A name that is hard to hear

"Adrien" is two syllables and shares its opening with ordinary speech, which
makes it a slightly harder wake word than something like "hey Jarvis". If false
triggers stay stubborn after tuning:

- Train on **"hey Adrien"** instead. The extra syllable is a large accuracy
  win — it is why almost every commercial assistant uses a two-word phrase.
- Feed the notebook more negative data drawn from your own environment (the
  notebook has a section for this).

## Which model is running?

```bash
python -m adrien doctor        # names the model and where it came from
python -m adrien status
```

The startup log says it too, at WARNING level while the fallback is in use — it
is meant to be slightly annoying, so the fallback does not quietly become
permanent.
