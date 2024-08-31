# Vocal

A vocal source separation


```
pip install vocal
```

```python
import librosa
import soundfile as sf
from vocal import get_model, separate_vocal

device = "cuda" # or cpu
audio, sr = librosa.load("audio.wav", sr=44100, mono=False)
model = get_model(device) # download model from HF
audio_data = separate_vocal(model, audio, device, silent=False)
sf.write("vocal.mp3", format="MP3", data=audio_data.T, samplerate=sr)
```


## CLI

Sinlge file

```shell
vocali -i audio.mp3 -o output.mp3
```

Folder

```shell
vocali -i audio/ -o output --recursive
```