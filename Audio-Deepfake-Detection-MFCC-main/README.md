# 🎵 Audio Deepfake Detection System

An intelligent audio deepfake detection system using **MFCC (Mel-Frequency Cepstral Coefficients)** features and an **SVM (Support Vector Machine)** classifier to distinguish between genuine human speech and AI-generated audio.

---

## 🎯 Features

- Detects **AI-generated speech** vs. **real human voices**
- Achieved **66.7% accuracy** on mixed audio datasets
- **Real-time analysis** via CLI and web interface
- Supports **WAV & MP3** audio formats
- Detects **AI voice clones** of the same speaker

---

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- 500MB free storage
- Web browser (for web interface)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ManverdhanSharma/Audio-Deepfake-Detection-MFCC.git
   cd Audio-Deepfake-Detection-MFCC
Create virtual environment:

Windows

bash
Copy code
python -m venv venv
venv\Scripts\activate
macOS/Linux

bash
Copy code
python3 -m venv venv
source venv/bin/activate
Install dependencies:

bash
Copy code
pip install librosa scikit-learn numpy pandas matplotlib seaborn flask joblib
Prepare audio files:

Place real voices in real_audio/

Place AI-generated voices in deepfake_audio/

Need at least 2 files per folder

Train the model:

bash
Copy code
python main.py
Launch web interface:

bash
Copy code
python app.py
Then open: http://localhost:5000

📁 Project Structure
bash
Copy code
├── main.py          # Training script + CLI testing
├── app.py           # Flask web application
├── real_audio/      # Genuine speech samples
├── deepfake_audio/  # AI-generated speech samples
├── svm_model.pkl    # Trained SVM model
└── scaler.pkl       # Feature scaler
🎮 Usage
Command Line
bash
Copy code
python main.py
Follow prompts to test audio files.

Web Interface
bash
Copy code
python app.py
Upload files at http://localhost:5000

🔧 How It Works
MFCC Extraction – Extracts 13 coefficients per audio sample

SVM Classification – Linear SVM for binary classification

Real-time Prediction – Labels audio as GENUINE or DEEPFAKE

Confidence Score – Displays prediction confidence

📊 Performance
Dataset: 5 real voices + 8 AI-generated voices

Accuracy: 66.7% overall

Individual Tests: 100% correct classification in limited tests

Capability: Detects AI-cloned voices

🛠️ Troubleshooting
No audio files found → Add WAV/MP3 to real_audio/ and deepfake_audio/

Import errors → Run pip install librosa scikit-learn flask

Web app not loading → Check http://localhost:5000

Low accuracy → Add more training samples (10+ per class recommended)

💡 Tech Stack
Audio Processing: librosa

Machine Learning: scikit-learn (SVM)

Web Framework: Flask

Feature Extraction: MFCC

Visualization: matplotlib, seaborn

👨‍💻 Author
Manverdhan Sharma

🔗 GitHub: @ManverdhanSharma

📄 License
MIT License – free to use and modify.

🏆 Highlights
✅ Machine Learning (MFCC + SVM)
✅ Full-Stack ML Deployment (CLI + Flask Web App)
✅ Audio Signal Processing
✅ AI Voice Clone Detection
