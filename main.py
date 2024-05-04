import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import librosa
import librosa.display
import soundfile as sf

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchaudio
from torchaudio import transforms

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras import layers, models

# Custom dataset class
class AudioDataset(Dataset):
    def __init__(self, audio_files, labels, transform=None):
        self.audio_files = audio_files
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio, sr = torchaudio.load(self.audio_files[idx])
        label = self.labels[idx]
        if self.transform:
            audio = self.transform(audio)
        return audio, label

# Define the CNN
class AudioClassifier(nn.Module):
    def __init__(self, num_classes):
        super(AudioClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 64 * 16 * 16)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load data
audio_files = ["path/to/audio1.wav", "path/to/audio2.wav"]
labels = [0, 1]  # Example labels
transform = nn.Sequential(MelSpectrogram(), AmplitudeToDB())

dataset = AudioDataset(audio_files, labels, transform)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Initialize model, loss, optimizer
model = AudioClassifier(num_classes=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(5):
    for audio, label in dataloader:
        output = model(audio)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

print("Training complete.")

# Evaluation
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for audio, label in dataloader:
        output = model(audio)
        _, predicted = torch.max(output.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()

    print(f"Accuracy: {100 * correct / total}%")