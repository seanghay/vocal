import torch
import numpy as np
from tqdm import tqdm


def separate_vocal(
  model,
  mix,
  device,
  n_fft=6144,
  chunks=30,
  sample_rate=44100,
  dim_f=3072,
  hop=1024,
  silent=True,
):
  # Pre-compute constants
  audio_chunk_size = chunks * sample_rate
  dim_t = 2**8
  dim_c = 4
  chunk_size = hop * (dim_t - 1)
  n_bins = n_fft // 2 + 1

  # Move window to GPU once
  window = torch.hann_window(window_length=n_fft, periodic=True).to(device)

  # Pre-compute frequency padding
  out_c = dim_c
  _freq_pad = torch.zeros([1, out_c, n_bins - dim_f, dim_t], device=device)

  # Convert input to correct format
  if mix.ndim == 1:
    mix = np.asfortranarray([mix, mix])

  # Convert mix to torch tensor and move to GPU
  mix = torch.from_numpy(mix).to(device)

  margin = sample_rate if sample_rate < audio_chunk_size else audio_chunk_size
  samples = mix.shape[-1]

  if chunks == 0 or samples < audio_chunk_size:
    audio_chunk_size = samples

  # Pre-allocate chunks on GPU
  chunk_samples = []
  for skip in range(0, samples, audio_chunk_size):
    s_margin = 0 if skip == 0 else margin
    end = min(skip + audio_chunk_size + margin, samples)
    start = skip - s_margin
    chunk_samples.append(mix[:, start:end])
    if end == samples:
      break

  margin_size = margin
  chunked_sources = []

  with tqdm(total=len(chunk_samples), disable=silent, ascii=True) as pbar:
    for cmix_position, cmix in enumerate(chunk_samples):
      n_sample = cmix.shape[1]
      trim = n_fft // 2
      gen_size = chunk_size - 2 * trim
      pad = gen_size - n_sample % gen_size

      # Perform padding on GPU
      mix_p = torch.cat(
        [
          torch.zeros(2, trim, device=device),
          cmix,
          torch.zeros(2, pad, device=device),
          torch.zeros(2, trim, device=device),
        ],
        dim=1,
      )

      # Process waves in batches
      mix_waves = []
      i = 0
      while i < n_sample + pad:
        waves = mix_p[:, i : i + chunk_size]
        mix_waves.append(waves)
        i += gen_size

      # Stack waves efficiently
      mix_waves = torch.stack(mix_waves)

      with torch.no_grad():
        # Process in a single forward pass
        x = mix_waves.reshape(-1, chunk_size)

        # Perform STFT
        x = torch.stft(
          x,
          n_fft=n_fft,
          hop_length=hop,
          window=window,
          center=True,
          return_complex=True,
        )
        x = torch.view_as_real(x)
        x = x.permute(0, 3, 1, 2)

        # Reshape efficiently
        x = x.reshape(-1, 2, 2, n_bins, dim_t).reshape(-1, dim_c, n_bins, dim_t)
        x = x[:, :, :dim_f]

        # Model inference with memory optimization
        spec_pred = (-model(-x) + model(x)) * 0.5

        # Post-processing on GPU
        x = torch.cat(
          [spec_pred, _freq_pad.expand(spec_pred.shape[0], -1, -1, -1)], dim=2
        )
        c = 2
        x = x.reshape(-1, c, 2, n_bins, dim_t).reshape(-1, 2, n_bins, dim_t)
        x = x.permute(0, 2, 3, 1).contiguous()

        # Inverse STFT
        x = torch.view_as_complex(x)
        x = torch.istft(x, n_fft=n_fft, hop_length=hop, window=window, center=True)
        x = x.reshape(-1, c, chunk_size)

        # Move to CPU only at the end
        tar_waves = x.cpu()

        tar_signal = (
          tar_waves[:, :, trim:-trim].transpose(0, 1).reshape(2, -1).numpy()[:, :-pad]
        )

        start = 0 if cmix_position == 0 else margin_size
        end = None if cmix_position == len(chunk_samples) - 1 else -margin_size
        if margin_size == 0:
          end = None

      chunked_sources.append([tar_signal[:, start:end]])
      pbar.update()

  return np.concatenate(chunked_sources, axis=-1)[0]


def get_model(device: str):
  from huggingface_hub import hf_hub_download

  device = device.lower()
  variant = "cuda" if device == "mps" else device
  
  return torch.jit.load(
    hf_hub_download("seanghay/vocalfile", f"UVR-MDX-NET-Voc_FT.{variant}.pt"),
    map_location=device if device == "mps" else None
  ).to(device)  # Move model to GPU immediately