import os.path

from datasets import load_dataset
import librosa
import soundfile as sf
import noisereduce as nr

dataset = load_dataset("DORI-SRKW/DORI-Orcasound")

input_dir = "input_audio"
output_dir = "cleaned_audio"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def clean_audio(input_file, outputfile):
    y, sr = librosa.load(input_file, sr=None)
    y_resample = librosa.resample(y, sr, 16000)
    y_deNoised = nr.reduce_noise(y_resample)
    sf.write(outputfile, y_deNoised, samplerate=16000)


for filename in os.listdir(input_dir):
    if filename.endswith(".flac"):
        input_file = os.path.join(input_dir, filename)
        output_file = os.path.join(output_dir, filename)
        print("Processing: ", filename)
        clean_audio(input_file, output_file)

print("All audio files processed.")
