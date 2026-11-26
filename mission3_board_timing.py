import numpy as np
from team_audio import get_clip, clip_list
from board_link import Board

FS = 16000

# Use ten 1-second clips to make a 10-second test signal.
clips = clip_list(8)[:10]
audio = np.concatenate([
    get_clip(8, clip_id, fs=FS, antialias=True)
    for clip_id, label in clips
])

# Convert float audio (-1..1) to little-endian int16 bytes.
payload = np.round(np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()

print("Audio seconds:", len(audio) / FS)
print("Payload bytes:", len(payload))

board = Board("COM5", baud=921600)

result = board.stream(
    payload,
    fs=FS,
    fmt="i16",
    scale=1.0,
    win=480,
    hop=320,
    nfft=512,
    mels=40,
    mfcc=0,
    feat=True,
    store="none",
    speed=1.0
)

compute = np.asarray(result["compute_us"])
lag = np.asarray(result["lag_s"])

print("Result keys:", result.keys())
print("Board summary:", result.get("board"))
print("Compute entries:", len(compute))
print("Mean compute (ms):", compute.mean() / 1000)
print("Max compute (ms):", compute.max() / 1000)
print("Final lag (s):", lag[-1] if len(lag) else "no lag data")
print("Lost features:", result.get("lost_features"))