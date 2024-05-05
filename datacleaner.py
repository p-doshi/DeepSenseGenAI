import os.path

from datasets import load_dataset
import librosa
import soundfile as sf
import noisereduce as nr

dataset = load_dataset("DORI-SRKW/DORI-Orcasound")

input_dir = "downloaded_files"
output_dir = "cleaned_audio"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def clean_audio(input_file, output_file):
    # Load the audio file
    y, sr = librosa.load(input_file, sr=None)
    # Resample the audio to 16000 Hz
    y_resample = librosa.resample(y, orig_sr=sr, target_sr=16000)
    # Reduce noise from the resampled audio
    y_deNoised = nr.reduce_noise(y=y_resample, sr=16000)
    # Write the cleaned audio to a file
    sf.write(output_file, y_deNoised, samplerate=16000)



for filename in os.listdir(input_dir):
    if filename.endswith(".flac"):
        input_file = os.path.join(input_dir, filename)
        output_file = os.path.join(output_dir, filename)
        print("Processing: ", filename)
        clean_audio(input_file, output_file)

print("All audio files processed.")
