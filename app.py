import os
import numpy as np
import pickle
import librosa
import re
import uuid
import imageio_ffmpeg
import noisereduce as nr
import torch
from google import genai
from transformers import pipeline
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# FFmpeg
os.environ["PATH"] += os.pathsep + os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())

# CONFIG 
SR = 16000
N_MELS = 128
MAX_FRAMES = 200
MAX_TOKENS = 80
AUDIO_UPLOAD_FOLDER = "uploads"
os.makedirs(AUDIO_UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = "model/best_late_with_cw.keras"
TOKENIZER_PATH = "model/tokenizer1.pkl"
ASR_MODEL_ID = "Showrov5843/showrov_azam_5843_bengali-whisper-medium"

# GEMINI SETUP 
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your-key-here")

client = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options={'api_version': 'v1'} 
)

def hybrid_fusion(tensors):
    p_f, p_a, p_t = tensors
    return 0.4 * p_f + 0.3 * p_a + 0.3 * p_t

# CLASSIFICATION 
model = load_model(MODEL_PATH, custom_objects={"hybrid_fusion": hybrid_fusion})
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

# WHISPER ASR 
print("Loading Whisper ASR Pipeline...")
device = 0 if torch.cuda.is_available() else -1
try:
    asr_pipeline = pipeline(
        "automatic-speech-recognition", 
        model=ASR_MODEL_ID,
        device=device
    )
except Exception as e:
    print(f"Error loading Whisper model: {e}")
    asr_pipeline = None

#regional to standard conversion using Gemini
def convert_regional_to_standard(regional_text):
    if not regional_text or not GEMINI_API_KEY:
        return regional_text
    
    prompt = (
        f"আপনি একজন অভিজ্ঞ বাংলা ভাষাবিদ। নিচের আঞ্চলিক বাংলা বাক্যটিকে "
        f"প্রমিত বাংলা (চলিত ভাষা)-তে রূপান্তর করুন। "
        f"শুধুমাত্র রূপান্তরিত বাংলা টেক্সটটুকু আউটপুট হিসেবে দিন।\n\n"
        f"আঞ্চলিক বাক্য: {regional_text}"
    )
    
    try:
        response = client.models.generate_content(
            model='gemini-2.0-flash', 
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        print(f"Gemini Error: {e}")
        return regional_text

# Text and audio processing
bangla_stopwords = set(["আমি","আমরা","তুমি","আপনি","সে","তিনি","তারা","এটি","সেটা","কোন","কোনো","কি","কী","কেন","যে","যিনি","এবং","ও","বা","তো","তাই"])
punct_pattern = re.compile(r"[^০-৯া-ৣঁংঃঀ-৿\s]+")

def clean_text(text):
    text = str(text)
    text = punct_pattern.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [w for w in text.split() if w not in bangla_stopwords]
    return " ".join(tokens)

def text_to_sequence(text):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    return pad_sequences(seq, maxlen=MAX_TOKENS, padding="post")

def load_clean_audio(filepath):
    try:
        y, sr = librosa.load(filepath, sr=SR)
    except Exception as e:
        print(f"Error loading audio file: {e}")
        y, sr = np.zeros(SR * 2), SR
    
    y = nr.reduce_noise(y=y, sr=sr)
    y, _ = librosa.effects.trim(y, top_db=25)
    intervals = librosa.effects.split(y, top_db=30)
    
    if len(intervals) > 0:
        y = np.concatenate([y[s:e] for s, e in intervals])
    if len(y) == 0:
        y = np.zeros(SR * 2) 
    return y, sr

def get_melspec(y, sr):
    melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    melspec_db = librosa.power_to_db(melspec, ref=np.max).T
    if melspec_db.shape[0] < MAX_FRAMES:
        pad_w = MAX_FRAMES - melspec_db.shape[0]
        melspec_db = np.pad(melspec_db, ((0, pad_w), (0, 0)), mode="constant")
    else:
        melspec_db = melspec_db[:MAX_FRAMES, :]
    return melspec_db.astype("float32")

CLASS_NAMES = ["চোখের সমস্যা", "ত্বকের সমস্যা", "ব্যথা", "মাথার সমস্যা", "শারীরিক দুর্বলতা", "শ্বাসকষ্ট", "সংক্রামিত ক্ষত"]

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    typed_text = request.form.get("text", "").strip()
    audio_file = request.files.get("audio")

    if typed_text == "" and (not audio_file or audio_file.filename == ""):
        return jsonify({"prediction": "❌ কোনো ইনপুট পাওয়া যায়নি!"})

    final_text_input = typed_text

    # audio file handling and ASR
    if audio_file and audio_file.filename != "":
        filename = str(uuid.uuid4()) + ".wav"
        filepath = os.path.join(AUDIO_UPLOAD_FOLDER, filename)
        audio_file.save(filepath)
        y_clean, sr_clean = load_clean_audio(filepath)
        
        # whisper speech-to-text
        if asr_pipeline:
            try:
                asr_output = asr_pipeline(filepath, chunk_length_s=30)
                regional_text = asr_output['text'].strip()
                
                # Gemini regional to standard conversion
                standard_text = convert_regional_to_standard(regional_text)
                print(f"Whisper (Regional): {regional_text}")
                print(f"Gemini (Standard): {standard_text}")
                
                # if user didn't type anything, use ASR+Gemini output
                if not typed_text:
                    final_text_input = standard_text
            except Exception as e:
                print(f"ASR/Gemini Error: {e}")

        audio_feat = np.expand_dims(get_melspec(y_clean, sr_clean), axis=0)
    else:
        audio_feat = np.zeros((1, MAX_FRAMES, N_MELS), dtype="float32")

    # text processing
    text_feat = text_to_sequence(final_text_input) if final_text_input else np.zeros((1, MAX_TOKENS), dtype="float32")

    # model prediction
    if not model:
        final_class_name = "Model Error: Classifier not loaded."
    else:
        # model prediction and confidence check
        pred = model.predict([audio_feat, text_feat])[0] 
        max_prob = np.max(pred)
        has_bengali = bool(re.search(r'[\u0980-\u09FF]', final_text_input))
        
        if (final_text_input != "" and not has_bengali) or max_prob < 0.15:
            final_class_name = "আপনার তথ্যে ভুল আছে! শুধুমাত্র স্বাস্থ্য সম্পর্কিত সঠিক তথ্য দিন।"
        else:
            final_class_name = CLASS_NAMES[np.argmax(pred)]

    return jsonify({
        "prediction": final_class_name,
        "transcription": final_text_input 
    })

if __name__ == "__main__":
    app.run(debug=True)