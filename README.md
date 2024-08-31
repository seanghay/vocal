# Vocal

A vocal source separation


```python
from vocal import get_model, separate_vocal

device = "cpu"
audio, sr = librosa.load("audio.wav", mono=False, sr=44100)
model = get_model(device)
audio_data = separate_vocal(model, audio, device)
```