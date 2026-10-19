import gc
import hashlib
import os
import logging
import warnings

import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
from tqdm import tqdm

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
stem_naming = {'Vocals': 'Instrumental', 'Other': 'Instruments', 'Instrumental': 'Vocals', 'Drums': 'Drumless', 'Bass': 'Bassless'}


class MDXModel:
    def __init__(self, device, dim_f, dim_t, n_fft, hop=1024, stem_name=None, compensation=1.000):
        self.dim_f = dim_f
        self.dim_t = dim_t
        self.dim_c = 4
        self.n_fft = n_fft
        self.hop = hop
        self.stem_name = stem_name
        self.compensation = compensation

        self.n_bins = self.n_fft // 2 + 1
        self.chunk_size = hop * (self.dim_t - 1)
        self.window = torch.hann_window(window_length=self.n_fft, periodic=True).to(device)

        out_c = self.dim_c

        self.freq_pad = torch.zeros([1, out_c, self.n_bins - self.dim_f, self.dim_t]).to(device)

    def stft(self, x):
        x = x.reshape([-1, self.chunk_size])
        x = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop, window=self.window, center=True, return_complex=True)
        x = torch.view_as_real(x)
        x = x.permute([0, 3, 1, 2])
        x = x.reshape([-1, 2, 2, self.n_bins, self.dim_t]).reshape([-1, 4, self.n_bins, self.dim_t])
        return x[:, :, :self.dim_f]

    def istft(self, x, freq_pad=None):
        freq_pad = self.freq_pad.repeat([x.shape[0], 1, 1, 1]) if freq_pad is None else freq_pad
        x = torch.cat([x, freq_pad], -2)
        # c = 4*2 if self.target_name=='*' else 2
        x = x.reshape([-1, 2, 2, self.n_bins, self.dim_t]).reshape([-1, 2, self.n_bins, self.dim_t])
        x = x.permute([0, 2, 3, 1])
        x = x.contiguous()
        x = torch.view_as_complex(x)
        x = torch.istft(x, n_fft=self.n_fft, hop_length=self.hop, window=self.window, center=True)
        return x.reshape([-1, 2, self.chunk_size])


class MDX:
    DEFAULT_SR = 44100
    # Unit: seconds
    DEFAULT_CHUNK_SIZE = 0 * DEFAULT_SR
    DEFAULT_MARGIN_SIZE = 1 * DEFAULT_SR

    DEFAULT_PROCESSOR = 0

    def __init__(self, model_path: str, params: MDXModel, processor=DEFAULT_PROCESSOR):

        # Set the device and the provider (CPU or CUDA)
        self.device = torch.device(f'cuda:{processor}') if processor >= 0 else torch.device('cpu')
        self.provider = ['CUDAExecutionProvider'] if processor >= 0 else ['CPUExecutionProvider']

        self.model = params

        # Load the ONNX model using ONNX Runtime
        self.ort = ort.InferenceSession(model_path, providers=self.provider)
        # Preload the model for faster performance
        self.ort.run(None, {'input': torch.rand(1, 4, params.dim_f, params.dim_t).numpy()})
        self.process = lambda spec: self.ort.run(None, {'input': spec.cpu().numpy()})[0]

        self.prog = None

    @staticmethod
    def get_hash(model_path):
        try:
            with open(model_path, 'rb') as f:
                f.seek(- 10000 * 1024, 2)
                model_hash = hashlib.md5(f.read()).hexdigest()
        except:
            model_hash = hashlib.md5(open(model_path, 'rb').read()).hexdigest()

        return model_hash

    @staticmethod
    def segment(wave, combine=True, chunk_size=DEFAULT_CHUNK_SIZE, margin_size=DEFAULT_MARGIN_SIZE):
        """
        Segment or join segmented wave array

        Args:
            wave: (np.array) Wave array to be segmented or joined
            combine: (bool) If True, combines segmented wave array. If False, segments wave array.
            chunk_size: (int) Size of each segment (in samples)
            margin_size: (int) Size of margin between segments (in samples)

        Returns:
            numpy array: Segmented or joined wave array
        """

        if combine:
            processed_wave = None  # Initializing as None instead of [] for later numpy array concatenation
            for segment_count, segment in enumerate(wave):
                start = 0 if segment_count == 0 else margin_size
                end = None if segment_count == len(wave) - 1 else -margin_size
                if margin_size == 0:
                    end = None
                if processed_wave is None:  # Create array for first segment
                    processed_wave = segment[:, start:end]
                else:  # Concatenate to existing array for subsequent segments
                    processed_wave = np.concatenate((processed_wave, segment[:, start:end]), axis=-1)

        else:
            processed_wave = []
            sample_count = wave.shape[-1]

            if chunk_size <= 0 or chunk_size > sample_count:
                chunk_size = sample_count

            if margin_size > chunk_size:
                margin_size = chunk_size

            for segment_count, skip in enumerate(range(0, sample_count, chunk_size)):

                margin = 0 if segment_count == 0 else margin_size
                end = min(skip + chunk_size + margin_size, sample_count)
                start = skip - margin

                cut = wave[:, start:end].copy()
                processed_wave.append(cut)

                if end == sample_count:
                    break

        return processed_wave

    def pad_wave(self, wave):
        """
        Pad the wave array to match the required chunk size

        Args:
            wave: (np.array) Wave array to be padded

        Returns:
            tuple: (padded_wave, pad, trim)
                - padded_wave: Padded wave array
                - pad: Number of samples that were padded
                - trim: Number of samples that were trimmed
        """
        n_sample = wave.shape[1]
        trim = self.model.n_fft // 2
        gen_size = self.model.chunk_size - 2 * trim
        pad = gen_size - n_sample % gen_size

        # Padded wave
        wave_p = np.concatenate((np.zeros((2, trim)), wave, np.zeros((2, pad)), np.zeros((2, trim))), 1)

        # Keep the track in host memory; transfer only the current window.
        mix_waves = (
            wave_p[:, i:i + self.model.chunk_size]
            for i in range(0, n_sample + pad, gen_size)
        )

        return mix_waves, pad, trim

    def _process_wave(self, mix_waves, trim, pad):
        """
        Process CPU windows sequentially on the calling thread.

        Args:
            mix_waves: Iterator of CPU numpy windows to be processed
            trim: (int) Number of samples trimmed during padding
            pad: (int) Number of samples padded during padding

        Returns:
            numpy array: Processed wave segment
        """
        with torch.no_grad():
            pw = []
            for mix_wave in mix_waves:
                mix_wave = torch.as_tensor(
                    np.asarray(mix_wave, dtype=np.float32)
                ).unsqueeze(0).to(self.device)
                self.prog.update()
                spec = self.model.stft(mix_wave)
                processed_spec = torch.as_tensor(self.process(spec)).to(self.device)
                processed_wav = self.model.istft(processed_spec)
                processed_wav = processed_wav[:, :, trim:-trim].transpose(0, 1).reshape(2, -1).cpu().numpy()
                pw.append(processed_wav)
                del mix_wave, spec, processed_spec
        processed_signal = np.concatenate(pw, axis=-1)
        if pad:
            processed_signal = processed_signal[:, :-pad]
        return processed_signal

    def process_wave(self, wave: np.array, mt_threads=1):
        """
        Process a stereo track with one inference window on the GPU at a time.

        Args:
            wave: (np.array) Wave array to be processed
            mt_threads: Legacy compatibility argument; inference is sequential

        Returns:
            numpy array: Processed wave array
        """
        if wave.ndim != 2 or wave.shape[0] != 2 or wave.shape[1] == 0:
            raise ValueError("Audio must contain nonempty stereo samples")
        mix_waves, pad, trim = self.pad_wave(wave)
        gen_size = self.model.chunk_size - 2 * trim
        self.prog = tqdm(total=(wave.shape[1] + pad) // gen_size)
        try:
            # Keep mt_threads accepted for older callers, but execute inference
            # on this thread so the real exception and traceback reach RunPod.
            return self._process_wave(mix_waves, trim, pad)
        finally:
            self.prog.close()


def run_mdx(model_params, output_dir, model_path, filename, exclude_main=False, exclude_inversion=False, suffix=None, invert_suffix=None, denoise=False, keep_orig=True, m_threads=2):
    wave, sr = librosa.load(filename, mono=False, sr=44100)
    if wave.size == 0 or not np.isfinite(wave).all():
        raise ValueError("Audio is empty or contains invalid samples")
    if wave.ndim == 1:
        wave = np.stack((wave, wave))
    if wave.ndim != 2 or wave.shape[0] != 2:
        raise ValueError("Audio must have one or two channels")
    peak = float(np.max(np.abs(wave)))
    if peak == 0:
        raise ValueError("Audio contains only silence")

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    logger.info("MDX model=%s input=%s duration=%.2fs device=%s",
                os.path.basename(model_path), os.path.basename(filename),
                wave.shape[-1] / sr, device)

    model_hash = MDX.get_hash(model_path)
    mp = model_params.get(model_hash)
    if not mp:
        raise ValueError(f"Unknown MDX model parameters: {os.path.basename(model_path)}")
    model = MDXModel(
        device,
        dim_f=mp["mdx_dim_f_set"],
        dim_t=2 ** mp["mdx_dim_t_set"],
        n_fft=mp["mdx_n_fft_scale_set"],
        stem_name=mp["primary_stem"],
        compensation=mp["compensate"]
    )

    mdx_sess = MDX(model_path, model, processor=0 if device.type == "cuda" else -1)
    # normalizing input wave gives better output
    wave /= peak
    try:
        if denoise:
            wave_processed = -(mdx_sess.process_wave(-wave)) + (mdx_sess.process_wave(wave))
            wave_processed *= 0.5
        else:
            wave_processed = mdx_sess.process_wave(wave)
    except Exception:
        logger.exception("MDX inference failed: model=%s device=%s",
                         os.path.basename(model_path), device)
        raise
    finally:
        del mdx_sess
        gc.collect()
        if device.type == "cuda":
            logger.info("MDX GPU allocated=%d reserved=%d peak=%d",
                        torch.cuda.memory_allocated(), torch.cuda.memory_reserved(),
                        torch.cuda.max_memory_allocated())
            torch.cuda.empty_cache()
    # return to previous peak
    wave_processed *= peak
    stem_name = model.stem_name if suffix is None else suffix

    main_filepath = None
    if not exclude_main:
        main_filepath = os.path.join(output_dir, f"{os.path.basename(os.path.splitext(filename)[0])}_{stem_name}.wav")
        sf.write(main_filepath, wave_processed.T, sr)

    invert_filepath = None
    if not exclude_inversion:
        diff_stem_name = stem_naming.get(stem_name) if invert_suffix is None else invert_suffix
        stem_name = f"{stem_name}_diff" if diff_stem_name is None else diff_stem_name
        invert_filepath = os.path.join(output_dir, f"{os.path.basename(os.path.splitext(filename)[0])}_{stem_name}.wav")
        sf.write(invert_filepath, (-wave_processed.T * model.compensation) + wave.T, sr)

    if not keep_orig:
        os.remove(filename)

    del wave_processed, wave
    gc.collect()
    return main_filepath, invert_filepath
