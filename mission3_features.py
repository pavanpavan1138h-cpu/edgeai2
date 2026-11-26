import numpy as np
from team_audio import get_clip
import matplotlib.pyplot as plt
from team_audio import get_clip, clip_list, CLASSES

def hz_to_mel(f):
    return 2595 * np.log10(1 + f / 700)


def mel_to_hz(m):
    return 700 * (10 ** (m / 2595) - 1)

def features(x, fs, win, hop, nfft, mels, mfcc=None):
    """
    x: audio samples
    fs: sample rate
    win: frame length in samples
    hop: frame step in samples
    nfft: FFT size
    mels: number of mel bands
    mfcc: number of MFCC coefficients, or None
    """

    x = np.asarray(x, dtype=np.float32)

    if len(x) < win:
        raise ValueError("Audio is shorter than one frame.")

    frames = np.stack([
        x[start:start + win]
        for start in range(0, len(x) - win + 1, hop)
    ])

    window = np.hanning(win)
    windowed_frames = frames * window

    spectrum = np.fft.rfft(windowed_frames, n=nfft, axis=1)
    power = (np.abs(spectrum) ** 2) / nfft

    # Convert evenly spaced mel points back to Hz.
    mel_edges = np.linspace(
        hz_to_mel(20),
        hz_to_mel(fs / 2),
        mels + 2
    )
    hz_edges = mel_to_hz(mel_edges)

    # Map the FFT bins to their corresponding frequencies.
    fft_freqs = np.arange(nfft // 2 + 1) * fs / nfft

    # Build one triangular filter per mel band.
    filters = np.zeros((mels, nfft // 2 + 1), dtype=np.float32)

    for i in range(mels):
        left, center, right = hz_edges[i:i + 3]

        rising = (fft_freqs - left) / (center - left)
        falling = (right - fft_freqs) / (right - center)

        filters[i] = np.maximum(
            0,
            np.minimum(rising, falling)
        )

    mel_power = power @ filters.T

    log_mel = 10 * np.log10(np.maximum(mel_power, 1e-10))

    # TODO 6: If mfcc is not None, apply an orthonormal DCT-II
    # to the log-mel values and keep the first mfcc coefficients.

    # TODO 7: Return the feature array.
    if mfcc is None:
        return log_mel

    n_mels = log_mel.shape[1]
    k = np.arange(mfcc)[:, None]
    n = np.arange(n_mels)[None, :]

    dct_matrix = np.cos(np.pi / n_mels * (n + 0.5) * k)
    dct_matrix[0] *= np.sqrt(1 / n_mels)

    if mfcc > 1:
        dct_matrix[1:] *= np.sqrt(2 / n_mels)

    return log_mel @ dct_matrix.T


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fs = 16000
    win = int(0.030 * fs)
    hop = int(0.020 * fs)
    nfft = 512
    mels = 40

    clips = clip_list(8)

    # Select the first clip listed for each class.
    first_clip_by_class = {}
    for clip_id, label in clips:
        if label not in first_clip_by_class:
            first_clip_by_class[label] = clip_id

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for ax, label in zip(axes.flat, CLASSES):
        clip_id = first_clip_by_class[label]
        x = get_clip(8, clip_id, fs=fs, antialias=True)

        log_mel = features(x, fs, win, hop, nfft, mels=mels)

        ax.imshow(
            log_mel.T,
            origin="lower",
            aspect="auto",
            extent=[0, len(x) / fs, 0, mels]
        )
        ax.set_title(f"{label} — clip {clip_id}")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Mel band")
    
    mfcc_features = features(x, fs, win, hop, nfft, mels=40, mfcc=13)
    print("MFCC shape:", mfcc_features.shape)

    fig.tight_layout()
    fig.savefig("mission3_logmel_six_classes.png", dpi=150)
    plt.show()