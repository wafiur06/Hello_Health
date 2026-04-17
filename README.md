Hello Health - Bengali Health Symptom Classifier
A Flask-based web application that classifies health symptoms in Bengali language using multimodal machine learning. The app supports both text and voice input, automatically transcribes speech using Whisper ASR, and converts regional Bengali dialects to standard Bengali using Google's Gemini AI.

Features
Multimodal Input: Accepts both text and audio input for symptom description

Bengali Language Support: Processes Bengali text and speech with dialect normalization

Automatic Speech Recognition: Uses Whisper model fine-tuned for Bengali speech

Symptom Classification: Classifies symptoms into 7 health categories:

চোখের সমস্যা (Eye problems)
ত্বকের সমস্যা (Skin problems)
ব্যথা (Pain)
মাথার সমস্যা (Head problems)
শারীরিক দুর্বলতা (Physical weakness)
শ্বাসকষ্ট (Breathing difficulties)
সংক্রামিত ক্ষত (Infected wounds)
Real-time Voice Recording: Web-based audio recording interface

Input Validation: Ensures health-related content and proper Bengali text

Technology Stack
Backend: Flask (Python)
Machine Learning:
TensorFlow/Keras for symptom classification
Hugging Face Transformers for Whisper ASR
Google Gemini AI for dialect conversion
Audio Processing: Librosa, noisereduce
Frontend: HTML5, CSS3, JavaScript, Bootstrap
Deployment: Vercel
