# 🩺 Hello_Health: Multimodal Health Classification

**Hello_Health** is an AI-powered health informatics application designed to bridge the gap between regional Bengali dialects and automated health diagnosis. It accepts both text and audio inputs, converts regional speech into standard Bengali, and classifies the health issue into specific medical categories.

## 🚀 Features

* **Multimodal Prediction:** Uses a hybrid fusion model (Audio + Text) to increase classification accuracy.
* **Regional-to-Standard Bengali:** Integrates **Gemini 2.0 Flash** to normalize regional dialects into standard Bengali (Cholitobhasha).
* **Speech-to-Text:** Utilizes a custom fine-tuned **Whisper Medium** model for high-accuracy Bengali ASR (Automatic Speech Recognition).
* **Audio Pre-processing:** Implements noise reduction and silence trimming for cleaner spectral analysis.
* **Confidence Guardrails:** Built-in validation to detect non-Bengali input or low-confidence medical data to ensure safety.

---

## 🛠️ Tech Stack

* **Frontend:** Flask (Jinja2 Templates)
* **Backend:** Python (Flask)
* **Deep Learning:** TensorFlow/Keras (Hybrid Model), PyTorch (Whisper)
* **NLP & LLM:** Google GenAI (Gemini API), Hugging Face Transformers
* **Signal Processing:** Librosa, Noisereduce, FFmpeg

---

## 📁 Project Structure

```text
├── model/
│   ├── best_late_with_cw.keras  # Trained hybrid fusion model
│   └── tokenizer1.pkl           # Text tokenizer
├── uploads/                     # Temporary storage for audio files
├── templates/
│   └── index.html               # Web UI
├── app.py                       # Main Flask application
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Prerequisites
Ensure you have **FFmpeg** installed on your system.
* **Ubuntu:** `sudo apt install ffmpeg`
* **Mac:** `brew install ffmpeg`

### 2. Clone the Repository
```bash
git clone https://github.com/wafiur06/Hello_Health.git
cd Hello_Health
```

### 3. Install Dependencies
```bash
pip install os numpy pickle librosa re uuid imageio_ffmpeg noisereduce torch flask tensorflow transformers google-genai
```

### 4. Environment Variables
Set your Gemini API key:
```bash
# Windows
set GEMINI_API_KEY=your_api_key_here
# Linux/Mac
export GEMINI_API_KEY=your_api_key_here
```

---

## 🩺 Classification Categories
The model is trained to identify the following **7 categories**:
1.  চোখের সমস্যা (Eye Problems)
2.  ত্বকের সমস্যা (Skin Problems)
3.  ব্যথা (Pain)
4.  মাথার সমস্যা (Head/Neurological)
5.  শারীরিক দুর্বলতা (Physical Weakness)
6.  শ্বাসকষ্ট (Breathing Difficulty)
7.  সংক্রামিত ক্ষত (Infected Wounds)

---

## 🛠️ Core Functions

### Hybrid Fusion Logic
The model uses a weighted fusion approach to combine Mel-spectrogram features and text sequences:
$$0.4 \times P_{feature} + 0.3 \times P_{audio} + 0.3 \times P_{text}$$

### Dialect Normalization
A specific prompt is used to guide Gemini in translating regional dialects:
> "আপনি একজন অভিজ্ঞ বাংলা ভাষাবিদ। নিচের আঞ্চলিক বাংলা বাক্যটিকে প্রমিত বাংলা (চলিত ভাষা)-তে রূপান্তর করুন।"

---

## 🚦 Usage
1. Run the application:
   ```bash
   python app.py
   ```
2. Open `http://127.0.0.1:5000` in your browser.
3. Upload a Bengali voice recording or type your symptoms.
4. View the standardized text and the predicted health category.

## 📄 License
Distributed under the MIT License. See `LICENSE` for more information.
