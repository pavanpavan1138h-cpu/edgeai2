import numpy as np
from team_audio import clip_list, get_clip
from board_link import Board

FS = 16000
WIN = 480
HOP = 320
REPEATS = 10

clip_id, label = clip_list(8)[0]
audio = get_clip(8, clip_id, fs=FS, antialias=True)

# Repeat the one-second clip ten times.
audio = np.tile(audio, REPEATS)

payload = (
    np.round(np.clip(audio, -1.0, 1.0) * 32767)
    .astype("<i2")
    .tobytes()
)

print("Clip:", clip_id, label)
print("Audio seconds:", len(audio) / FS)
print("Payload bytes:", len(payload))

board = Board("COM6", baud=921600)

result = board.stream(
    payload,
    fs=FS,
    fmt="i16",
    scale=1.0,
    win=WIN,
    hop=HOP,
    nfft=512,
    mels=40,
    mfcc=0,
    feat=False,
    store="none",
    speed=7.25,
)

print("Result keys:", result.keys())
print("Board summary:", result.get("board"))

compute = np.asarray(result.get("compute_us", []))
lag = np.asarray(result.get("lag_s", []))

if len(compute):
    print("Compute entries:", len(compute))
    print("Mean compute (ms):", compute.mean() / 1000)
    print("Max compute (ms):", compute.max() / 1000)

if len(lag):
    print("Final lag (s):", lag[-1])

print("Lost features:", result.get("lost_features"))

for key in (
    "expected_features",
    "packets_sent",
    "bytes_sent",
    "send_seconds",
    "audio_seconds",
    "bad_crc_back",
    "ready",
    "features",
    "frame_numbers",
):
    value = result.get(key)
    if key in ("features", "frame_numbers") and value is not None:
        value = np.asarray(value).shape
    print(f"{key}: {value}")