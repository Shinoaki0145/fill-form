import numpy as np
import soxr
from pathlib import Path
import shutil
import subprocess
import soundfile as sf


def load_audio_file(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError("Không tìm thấy file âm thanh")
    if path.stat().st_size > 50 * 1024 * 1024:
        raise ValueError("File âm thanh tối đa 50 MB")
    ingress = AudioIngress()
    frames = []
    try:
        source = sf.SoundFile(path)
    except RuntimeError:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise ValueError("Không đọc được định dạng audio này; cài ffmpeg hoặc dùng WAV/FLAC/MP3") from None
        try:
            result = subprocess.run(
                [ffmpeg, "-nostdin", "-v", "error", "-i", str(path.resolve()),
                 "-map", "0:a:0", "-t", "301", "-ac", "1", "-ar", "16000", "-f", "f32le", "pipe:1"],
                capture_output=True, timeout=60, check=False,
            )
        except subprocess.TimeoutExpired:
            raise ValueError("Giải mã audio quá lâu; thử file ngắn hơn") from None
        if result.returncode or not result.stdout:
            raise ValueError("File âm thanh lỗi hoặc định dạng không được hỗ trợ")
        audio = np.frombuffer(result.stdout, dtype="<f4")
        if len(audio) > 300 * 16000:
            raise ValueError("File âm thanh tối đa 5 phút")
        frames = ingress.push(16000, audio) + ingress.finish()
        rate, duration = 16000, len(audio) / 16000
    else:
        with source:
            if source.frames / source.samplerate > 300:
                raise ValueError("File âm thanh tối đa 5 phút")
            rate, duration = source.samplerate, source.frames / source.samplerate
            for block in source.blocks(blocksize=65536, dtype="float32", always_2d=True):
                frames.extend(ingress.push(rate, block))
            frames.extend(ingress.finish())
    if not frames:
        raise ValueError("File âm thanh rỗng")
    return frames, duration, rate


class AudioIngress:
    def __init__(self):
        self.rate = None
        self.resampler = None
        self.tail = np.empty(0, np.float32)
        self.finished = False

    def frames(self, wave):
        self.tail = np.concatenate((self.tail, wave))
        count = len(self.tail) // 512
        result = [x.copy() for x in self.tail[: count * 512].reshape(-1, 512)]
        self.tail = self.tail[count * 512 :]
        return result

    def push(self, rate, samples):
        if self.finished:
            raise ValueError("Recording đã đóng")
        if self.rate is not None and self.rate != rate:
            raise ValueError("Sample rate thay đổi giữa recording")
        x = np.asarray(samples)
        if np.issubdtype(x.dtype, np.integer):
            x = x.astype(np.float32) / max(
                abs(np.iinfo(x.dtype).min), np.iinfo(x.dtype).max
            )
        else:
            x = x.astype(np.float32)
        if x.ndim == 2:
            x = x.mean(axis=1)
        if x.ndim != 1 or not np.isfinite(x).all():
            raise ValueError("Audio không hợp lệ")
        if self.rate is None:
            self.rate = rate
            if rate != 16000:
                self.resampler = soxr.ResampleStream(rate, 16000, 1, dtype="float32")
        if self.resampler:
            x = self.resampler.resample_chunk(x)
        return self.frames(x)

    def finish(self):
        if self.finished:
            return []
        self.finished = True
        result = (
            self.frames(
                self.resampler.resample_chunk(np.empty(0, np.float32), last=True)
            )
            if self.resampler
            else []
        )
        if self.tail.size:
            result.append(np.pad(self.tail, (0, 512 - len(self.tail))))
            self.tail = np.empty(0, np.float32)
        return result
