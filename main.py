#!/usr/bin/env python3
import os
import sys
import subprocess
from vosk import Model, KaldiRecognizer
import wave
from gtts import gTTS

CALLER_ID = sys.argv[1]
BASE_DIR = "/home/asteriskvm/ivr-module"
TMP_DIR = "/tmp"

AUDIO_INPUT = f"{BASE_DIR}/tmp/caller_audio.wav"
AUDIO_OUTPUT = f"{TMP_DIR}/{CALLER_ID}_response.wav"
AUDIO_OUTPUT_ULAW = f"{TMP_DIR}/{CALLER_ID}_response.ulaw"
HISTORY_FILE = f"{TMP_DIR}/{CALLER_ID}_history.txt"

MODEL_PATH = "/home/asteriskvm/vosk-models/vosk-model-small-en-us-0.15"

# ------------------------
# Helper: STT (with chunking)
# ------------------------
def transcribe_audio(audio_path):
    # Resample audio to 16k
    resampled_path = audio_path.replace(".wav", "_16k.wav")
    subprocess.run(["ffmpeg", "-y", "-i", audio_path, "-ar", "16000", resampled_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    wf = wave.open(resampled_path, "rb")
    model = Model(MODEL_PATH)
    rec = KaldiRecognizer(model, wf.getframerate())

    transcript = ""
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            res = rec.Result()
            transcript += res
    transcript += rec.FinalResult()

    # Extract text only
    text = transcript.replace("\n", " ").replace("{", "").replace("}", "")
    text = text.split("text")[-1].replace(":", "").replace('"', '').strip()
    return text


# ------------------------
# Helper: LLM (Ollama Phi with context memory)
# ------------------------
def call_llm(prompt):
    # Load history
    history = ""
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            history = f.read()

    # Build prompt
    full_prompt = (
        "You are a helpful AI assistant who explains concepts in simple educational terms.\n"
        + history
        + f"\nUser: {prompt}\nAssistant:"
    )

    # Call Ollama
    result = subprocess.check_output(
        ["ollama", "run", "phi", full_prompt], text=True
    ).strip()

    # Save history
    with open(HISTORY_FILE, "w") as f:
        f.write(history + f"\nUser: {prompt}\nAssistant: {result}")

    return result


# ------------------------
# Helper: TTS
# ------------------------
def text_to_speech(text, output_path):
    if not text.strip():
        text = "I'm sorry, I didn't catch that. Could you repeat?"
    tts = gTTS(text=text, lang="en")
    tts.save(output_path)

    # Convert mp3->ulaw for Asterisk
    subprocess.run([
        "ffmpeg", "-y", "-i", output_path, "-ar", "8000", "-ac", "1", "-f", "mulaw", AUDIO_OUTPUT_ULAW
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ------------------------
# MAIN PIPELINE
# ------------------------
def main():
    print("[STT] Transcribing with Vosk...")
    user_text = transcribe_audio(AUDIO_INPUT)
    print(f"[User said]: {user_text}")

    if not user_text.strip():
        user_text = "No input detected"

    # Goodbye detection
    if any(word in user_text.lower() for word in ["bye", "goodbye", "exit", "quit", "thank you"]):
        response = "Goodbye! It was nice talking to you."
    else:
        print("[LLM] Calling Ollama Phi...")
        response = call_llm(user_text)

    print(f"[LLM Response]: {response}")

    print("[TTS] Generating with gTTS...")
    text_to_speech(response, AUDIO_OUTPUT)

    print(f"[DONE] Response ready: {AUDIO_OUTPUT_ULAW}")


if __name__ == "__main__":
    main()
