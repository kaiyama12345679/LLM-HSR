import whisper
import pyaudio


import pyaudio
import numpy as np
import whisper

# 音声録音の設定
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
THRESHOLD = 500  # 音量のしきい値
SILENCE_DURATION = 2  # 録音終了と判断する無音の持続時間（秒）

def is_silent(data_chunk):
    """データチャンクが無音かどうかを判定する"""
    audio_data = np.frombuffer(data_chunk, dtype=np.int16)
    return np.max(audio_data) < THRESHOLD

def record_audio():
    """音声を録音し、NumPy配列として返す"""
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("録音開始を待機中...")

    frames = []
    silence_count = 0
    recording = False

    while True:
        data = stream.read(CHUNK)
        if recording:
            frames.append(data)
            if is_silent(data):
                silence_count += 1
            else:
                silence_count = 0
            if silence_count > SILENCE_DURATION * (RATE / CHUNK):
                break
        elif not is_silent(data):
            print("音声検出、録音を開始します...")
            recording = True
            frames.append(data)

    print("録音終了")

    stream.stop_stream()
    stream.close()
    p.terminate()

    # NumPy配列に変換
    audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
    audio_data = np.copy(audio_data).astype(np.float32)
    return audio_data

if __name__ == "__main__":
    # 音声を録音
    audio_data = record_audio()

    # Whisperモデルをロード
    model = whisper.load_model("large")

    # 音声データをモデルに入力して文字起こし
    result = model.transcribe(audio_data, language="ja")

    # 結果を表示
    print(result["text"])