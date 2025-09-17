import soundfile  as sf 
import os
import shutil
import librosa
from pyannote.audio import Model, Inference
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import tempfile
import torch

def load_model():
    model = Model.from_pretrained("pyannote/embedding", 
                                )
    inference = Inference(model, window="whole")
    return inference

def pre_process(audio_array, sr):
    if sr != 16000:
        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)

    return audio_array

def load_audio(file_path):
    audio_array, sr = sf.read(file_path)
    audio_array = pre_process(audio_array, sr)
    return audio_array

def get_segments(audio_array, segment_length=0.020, sr=16000):
    segment_samples = int(segment_length * sr)
    total_samples = len(audio_array)
    segments = []
    for start in range(0, total_samples, segment_samples):
        end = min(start + segment_samples, total_samples)
        segments.append(audio_array[start:end])
        if end == total_samples:
            break
    return segments

def extract_embeddings(segments, inference):
    embeddings = []
    for segment in segments:
        if len(segment) < 16000:  # Skip segments shorter than 1 second
            continue
        segment_tensor = torch.tensor(segment, dtype=torch.float32).unsqueeze(0)
        audio_dict = {"waveform": segment_tensor, "sample_rate": 16000}
        embedding = inference(audio_dict)
        embeddings.append(embedding)
    return np.array(embeddings)

def generate_tsne_plot(embeddings):
    
    tsne = TSNE(n_components=2, perplexity=5)
    tsne_results = tsne.fit_transform(embeddings)
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1])
    plt.savefig("tsne_plot.png", bbox_inches='tight', dpi=300) 
    plt.close()

inference = load_model()
audio_path = "/Users/cmi_10128/Downloads/english_test.wav"
audio_array = load_audio(audio_path)
segments = get_segments(audio_array, segment_length=1, sr=16000)
embeddings = extract_embeddings(segments, inference)
generate_tsne_plot(embeddings)

