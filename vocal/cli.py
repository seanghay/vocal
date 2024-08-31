import torch
import librosa
import soundfile
import click
from tqdm import tqdm
from pathlib import Path
from vocal import get_model, separate_vocal
from typing import List


@click.command()
@click.option(
  "-i",
  "--input",
  type=click.Path(exists=True, dir_okay=True, file_okay=True, path_type=Path),
  required=True,
)
@click.option(
  "-o",
  "--output",
  type=click.Path(writable=True, path_type=Path),
  required=True,
)
@click.option("-r", "--recursive", default=False, help="Recursive", is_flag=True)
@click.option(
  "-d",
  "--device",
  default="cuda" if torch.cuda.is_available() else "cpu",
  help="Device",
)
@click.option(
  "-g",
  "--glob",
  help="Glob pattern",
  default=["*.wav", "*.mp3", "*.flac", "*.ogg"],
  multiple=True,
)
@click.option("-q", "--quiet", default=False, help="Quiet", is_flag=True)
@click.option("-c", "--chunk", type=click.INT, default=30, help="Audio chunk size")
def cli(
  input: Path,
  output: Path,
  device: str,
  recursive: bool,
  quiet: bool,
  glob: List[str],
  chunk: int,
):
  model = get_model(device)
  if input.is_dir():
    files = [
      file
      for pat in glob
      for file in (input.rglob(pat) if recursive else input.glob(pat))
    ]

    for file in tqdm(files, disable=quiet):
      target_file = output / file.relative_to(Path(input))
      target_file.parent.mkdir(exist_ok=True)
      audio, sr = librosa.load(file, sr=44100, mono=False)
      audio_data = separate_vocal(
        model, audio, device=device, silent=quiet, chunks=chunk
      )
      soundfile.write(target_file, audio_data.T, samplerate=sr)
    return

  audio, sr = librosa.load(input, sr=44100, mono=False)
  audio_data = separate_vocal(model, audio, device=device, silent=quiet, chunks=chunk)
  soundfile.write(output, audio_data.T, samplerate=sr)
