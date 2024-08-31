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
  audio_chunk_size = chunks * sample_rate
  dim_t = 2**8
  dim_c = 4
  chunk_size = hop * (dim_t - 1)
  n_bins = n_fft // 2 + 1
  window = torch.hann_window(window_length=n_fft, periodic=True).to(device)
  out_c = dim_c
  _freq_pad = torch.zeros([1, out_c, n_bins - dim_f, dim_t]).to(device)

  if mix.ndim == 1:
    mix = np.asfortranarray([mix, mix])

  margin = sample_rate
  samples = mix.shape[-1]

  if margin > audio_chunk_size:
    margin = audio_chunk_size

  if chunks == 0 or samples < audio_chunk_size:
    audio_chunk_size = samples

  counter = -1
  chunk_samples = []
  for skip in range(0, samples, audio_chunk_size):
    counter += 1
    s_margin = 0 if counter == 0 else margin
    end = min(skip + audio_chunk_size + margin, samples)
    start = skip - s_margin
    chunk_samples.append(mix[:, start:end].copy())

    if end == samples:
      break

  margin_size = margin
  chunked_sources = []

  with tqdm(total=len(chunk_samples), disable=silent) as pbar:
    for cmix_position, cmix in enumerate(chunk_samples):
      n_sample = cmix.shape[1]
      trim = n_fft // 2
      gen_size = chunk_size - 2 * trim
      pad = gen_size - n_sample % gen_size
      mix_p = np.concatenate(
        (np.zeros((2, trim)), cmix, np.zeros((2, pad)), np.zeros((2, trim))), 1
      )

      i = 0
      mix_waves = []

      while i < n_sample + pad:
        waves = np.array(mix_p[:, i : i + chunk_size])
        mix_waves.append(waves)
        i += gen_size

      mix_waves = torch.tensor(np.array(mix_waves), dtype=torch.float32).to(device)

      with torch.no_grad():
        x = mix_waves
        x = x.reshape([-1, chunk_size])
        x = torch.stft(
          x,
          n_fft=n_fft,
          hop_length=hop,
          window=window,
          center=True,
          return_complex=True,
        )
        x = torch.view_as_real(x)
        x = x.permute([0, 3, 1, 2])
        x = x.reshape([-1, 2, 2, n_bins, dim_t]).reshape([-1, dim_c, n_bins, dim_t])
        x = x[:, :, :dim_f]
        spec_pred = (-model(-x) * 0.5) + (model(x) * 0.5)
        x = spec_pred
        freq_pad = _freq_pad.repeat([x.shape[0], 1, 1, 1])
        x = torch.cat([x, freq_pad], -2)
        c = 2
        x = x.reshape([-1, c, 2, n_bins, dim_t]).reshape([-1, 2, n_bins, dim_t])
        x = x.permute([0, 2, 3, 1])
        x = x.contiguous()
        x = torch.view_as_complex(x)
        x = torch.istft(x, n_fft=n_fft, hop_length=hop, window=window, center=True)
        x = x.reshape([-1, c, chunk_size])
        tar_waves = x.reshape([-1, c, chunk_size]).cpu()

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
  return torch.jit.load(
    hf_hub_download("seanghay/vocalfile", f"UVR-MDX-NET-Voc_FT.{device}.pt")
  )
