import os
import numpy as np
import librosa
import torch
import torch.nn as nn
import tempfile
import subprocess
from scipy import stats

class AudioFeatureExtractor:
    """
    Extracts audio features using a frozen VGGish backbone.
    """
    def __init__(self, device='cpu', sample_rate=16000):
        self.device      = device
        self.sample_rate = sample_rate
        self._ffmpeg_exe = self._get_ffmpeg()

    def _get_ffmpeg(self):
        """
        Get ffmpeg binary from imageio_ffmpeg.
        No moviepy needed.
        """
        try:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
            print(f"  ✅ ffmpeg found: {exe}")
            return exe
        except ImportError:
            print("  ❌ imageio-ffmpeg not found")
            print("     Fix: pip install imageio-ffmpeg")
            return None

    def _extract_wav_from_video(self, video_path):
        """
        Extract audio using imageio-ffmpeg directly.
        No moviepy dependency at all.
        """
        if self._ffmpeg_exe is None:
            return None

        wav_path = None
        try:
            tmp      = tempfile.NamedTemporaryFile(
                suffix = '.wav',
                delete = False
            )
            wav_path = tmp.name
            tmp.close()

            cmd = [
                self._ffmpeg_exe,
                '-y',                       # overwrite output
                '-i',      video_path,      # input video
                '-vn',                      # skip video stream
                '-acodec', 'pcm_s16le',     # raw PCM
                '-ar',     '16000',         # 16kHz sample rate
                '-ac',     '1',             # mono
                wav_path                    # output wav
            ]

            result = subprocess.run(
                cmd,
                stdout  = subprocess.DEVNULL,
                stderr  = subprocess.PIPE,
                timeout = 60
            )

            if result.returncode != 0:
                if os.path.exists(wav_path):
                    os.unlink(wav_path)
                return None

            size = os.path.getsize(wav_path)
            if size < 1000:
                os.unlink(wav_path)
                return None

            return wav_path

        except Exception as e:
            if wav_path and os.path.exists(wav_path):
                try:
                    os.unlink(wav_path)
                except Exception:
                    pass
            return None

    def process_audio(self, video_path):
        """
        Extract 128-dim feature vector from video audio.
        Uses imageio-ffmpeg → wav → librosa pipeline.
        """
        wav_path = None
        try:
            # ── Step 1: Extract wav via ffmpeg ────────────────────
            wav_path = self._extract_wav_from_video(video_path)
            if wav_path is None:
                return None

            # ── Step 2: Load with librosa ──────────────────────────
            y, sr = librosa.load(
                wav_path,
                sr       = self.sample_rate,
                mono     = True,
                duration = 30
            )

            if len(y) < self.sample_rate * 0.5:
                return None

            features = []

            # ── MFCCs (80 features) ────────────────────────────────
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
            features.extend(np.mean(mfccs, axis=1).tolist())   # 40
            features.extend(np.std(mfccs,  axis=1).tolist())   # 40

            # ── Chroma (24 features) ───────────────────────────────
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features.extend(np.mean(chroma, axis=1).tolist())  # 12
            features.extend(np.std(chroma,  axis=1).tolist())  # 12

            # ── Spectral Features (6 features) ────────────────────
            features.append(float(np.mean(
                librosa.feature.spectral_centroid(y=y,  sr=sr))))
            features.append(float(np.mean(
                librosa.feature.spectral_bandwidth(y=y, sr=sr))))
            features.append(float(np.mean(
                librosa.feature.spectral_rolloff(y=y,   sr=sr))))
            features.append(float(np.mean(
                librosa.feature.spectral_flatness(y=y))))
            features.append(float(np.mean(
                librosa.feature.spectral_contrast(y=y,  sr=sr))))
            features.append(float(np.mean(
                librosa.feature.zero_crossing_rate(y))))

            # ── RMS Energy (2 features) ────────────────────────────
            rms = librosa.feature.rms(y=y)
            features.append(float(np.mean(rms)))
            features.append(float(np.std(rms)))

            # ── Tempo (1 feature) ──────────────────────────────────
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            features.append(
                float(tempo) if np.isscalar(tempo)
                else float(tempo[0])
            )

            # ── Mel Spectrogram Bands (15 features) ───────────────
            mel    = librosa.feature.melspectrogram(
                y=y, sr=sr, n_mels=15
            )
            mel_db = librosa.power_to_db(mel, ref=np.max)
            features.extend(np.mean(mel_db, axis=1).tolist())  # 15

            # ── Pad / Trim to exactly 128 ──────────────────────────
            features = np.array(features, dtype=np.float32)
            if len(features) < 128:
                features = np.pad(
                    features,
                    (0, 128 - len(features)),
                    mode = 'constant'
                )
            else:
                features = features[:128]

            # Normalize to [-1, 1]
            max_val = np.max(np.abs(features))
            if max_val > 0:
                features = features / max_val

            return features                         # (128,) ✓

        except Exception as e:
            return None

        finally:
            if wav_path and os.path.exists(wav_path):
                try:
                    os.unlink(wav_path)
                except Exception:
                    pass

    def analyze_voice_authenticity(self, video_path):
        """
        Analyze pause patterns using ffmpeg → wav → librosa.
        No moviepy needed.
        """
        wav_path = None
        try:
            wav_path = self._extract_wav_from_video(video_path)
            if wav_path is None:
                return {'authenticity_score': 0.5}

            y, sr = librosa.load(
                wav_path,
                sr       = self.sample_rate,
                mono     = True,
                duration = 30
            )

            if len(y) < sr * 0.5:
                return {'authenticity_score': 0.5}

            # ── Pause Pattern ──────────────────────────────────────
            rms     = librosa.feature.rms(
                y            = y,
                frame_length = 512,
                hop_length   = 256
            )[0]
            silence    = rms < (np.mean(rms) * 0.1)
            pause_lens = []
            count      = 0

            for s in silence:
                if s:
                    count += 1
                elif count > 0:
                    pause_lens.append(count)
                    count = 0

            if len(pause_lens) < 2:
                return {'authenticity_score': 0.5}

            pause_array  = np.array(pause_lens, dtype=np.float32)
            irregularity = min(
                float(np.std(pause_array)) /
                (float(np.mean(pause_array)) + 1e-6),
                1.0
            )

            # ── Pitch Variation ────────────────────────────────────
            pitches, mags = librosa.piptrack(y=y, sr=sr)
            pitch_vals    = pitches[mags > np.median(mags)]

            pitch_var = min(
                float(np.std(pitch_vals)) /
                (float(np.mean(pitch_vals)) + 1e-6),
                1.0
            ) if len(pitch_vals) > 10 else 0.5

            authenticity = (0.5 * irregularity) + (0.5 * pitch_var)
            return {
                'authenticity_score': round(float(authenticity), 4)
            }

        except Exception:
            return {'authenticity_score': 0.5}

        finally:
            if wav_path and os.path.exists(wav_path):
                try:
                    os.unlink(wav_path)
                except Exception:
                    pass

    def extract_temporal_features(self, y, sr):
        """Temporal consistency from loaded audio array."""
        try:
            rms      = librosa.feature.rms(y=y)[0]
            rms_std  = float(np.std(rms))
            rms_mean = float(np.mean(rms))

            consistency = 1.0 - min(
                rms_std / (rms_mean + 1e-6), 1.0
            )
            return {'temporal_consistency': round(consistency, 4)}

        except Exception:
            return {'temporal_consistency': 0.0}


class AdvancedAudioAnalyzer:

    def __init__(self, device='cpu'):
        self.device            = device
        self.feature_extractor = AudioFeatureExtractor(device=device)
        print("  ✅ AdvancedAudioAnalyzer ready (imageio-ffmpeg mode)")

    def extract_features(self, video_path):
        features = self.feature_extractor.process_audio(video_path)
        if features is None:
            return np.zeros(128, dtype=np.float32), False
        return features, True

    def analyze_authenticity(self, video_path):
        return self.feature_extractor.analyze_voice_authenticity(video_path)

    def analyze_voice_authenticity(self, video_path):
        return self.feature_extractor.analyze_voice_authenticity(video_path)
