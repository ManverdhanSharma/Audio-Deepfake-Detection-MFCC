from pydub import AudioSegment
import glob
import os

print("Converting MP3 files to WAV...")
mp3_files = glob.glob("deepfake_audio/*.mp3")

for mp3_file in mp3_files:
    try:
        wav_file = mp3_file.replace('.mp3', '.wav')
        audio = AudioSegment.from_mp3(mp3_file)
        audio.export(wav_file, format="wav")
        print(f" Converted: {os.path.basename(mp3_file)}  {os.path.basename(wav_file)}")
    except Exception as e:
        print(f" Error converting {mp3_file}: {e}")

print("Conversion complete!")
