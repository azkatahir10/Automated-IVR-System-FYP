# Automated IVR System (FYP)

This project is a fully automated IVR (Interactive Voice Response) system that handles incoming calls without human intervention. It uses open-source, local tools for real-time transcription (STT), language understanding (LLM), and response generation (TTS).

---

## 👨‍💻 Project Members

- Azka Tahir (bscs22066)
- Nayab Gohar (bscs22110)
- Kainat Umer (bscs22028)
- Aliha Tariq (bscs22146)

Supervisor: Dr. Ali Humayun

---

## 💻 Tech Stack (Offline / Free / Local)

| Component      | Tool Used         |
|----------------|------------------|
| Telephony      | Asterisk (on Ubuntu VM) |
| Speech-to-Text | [whisper.cpp](https://github.com/ggerganov/whisper.cpp) - small.en |
| Language Model | [Ollama](https://ollama.com) - `phi` model |
| Text-to-Speech | [Orpheus](https://github.com/Edresson/orpheus) or [StyleTTS2](https://github.com/yl4579/StyleTTS2) |
| Call Testing   | Linphone mobile app |

---

## 📁 Folder Structure

```

ivr-module/
│
├── asterisk/              # Asterisk config files (extensions.conf, etc.)
├── whisper.cpp/           # Local STT engine
├── orpheus-tts-local/     # Local TTS engine
├── tmp/                   # Temporary audio input/output files
├── venv/                  # Python virtual environment
├── .env                   # API keys and config (optional)
├── main.py                # Main IVR Python logic
├── run.sh                 # Script called from Asterisk
├── README.md              # You're here!

````

---

## ⚙️ Installation & Setup

### 1. 🧱 System Requirements

- Ubuntu VM (e.g. on VMware or VirtualBox)
- Python 3.10+
- Asterisk (installed and running)
- Git, ffmpeg, gcc, make, and other build tools

---

### 2. 🧰 Clone the Repo

```bash
git clone https://github.com/azkatahir10/IVR-System---FYP.git
cd IVR-System---FYP
````

---

### 3. 🧪 Set Up Python Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

### 4. 🧠 Set Up Ollama (LLM)

Install and start Ollama:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull phi
ollama run phi
```

Make sure it's running at:
`http://localhost:11434`

---

### 5. 🔊 Set Up whisper.cpp (STT)

```bash
cd whisper.cpp
make
```

No need to download model manually; it's pulled via script.

---

### 6. 🔈 Set Up TTS

If using Orpheus:

```bash
cd orpheus-tts-local
# Follow their README to set up and test locally
```

---

### 7. 📞 Configure Asterisk

Edit your Asterisk config:

```bash
sudo nano /etc/asterisk/extensions.conf
```

And insert something like this:

```ini
[default]
exten => 1000,1,Answer()
 same => n,Playback(hello)
 same => n,System(/home/asteriskvm/ivr-module/run.sh ${CALLERID(num)})
 same => n,Playback(/tmp/${CALLERID(num)}_response.wav)
 same => n,Hangup()
```

Restart Asterisk:

```bash
sudo systemctl restart asterisk
```

---

### 8. 📱 Test Call

* Connect Linphone (on same network).
* Call extension `1000`.
* Speak your query.
* Wait for LLM response via TTS.

---

## 🚫 Ignored Files

To keep your repo clean, the following are excluded via `.gitignore`:

```
venv/
tmp/
*.wav
.env
whisper.cpp/
orpheus-tts-local/
caller_audio.wav
output.wav
```

---

## 🔄 Pipeline Flow

1. Call received on Asterisk
2. Caller input recorded and saved
3. Audio passed to `whisper.cpp` (STT)
4. Transcription sent to Ollama (`phi`) for LLM processing
5. Response text synthesized via Orpheus or StyleTTS2
6. Response audio played back to caller
7. Call ends or loops

---

## 📌 References

* [whisper.cpp](https://github.com/ggerganov/whisper.cpp)
* [Ollama](https://ollama.com)
* [Asterisk](https://www.asterisk.org/)
* [Orpheus TTS](https://github.com/Edresson/orpheus)
* [StyleTTS2](https://github.com/yl4579/StyleTTS2)

---

## 💡 Future Improvements

* Improve response relevance via prompt engineering or fine-tuned LLMs
* Add multi-turn conversation memory
* Add multi-language support
* Replace Linphone with SIP trunk for real-world deployment

---

## 🙌 Acknowledgements

Thanks to open-source contributors behind the tools we’ve integrated!

---

Let me know if you'd like a PDF version or want this automatically committed into your GitHub repo.
```
