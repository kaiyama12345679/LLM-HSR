import whisper
import pyaudio


import pyaudio
import numpy as np
import whisper
import wave
import os

# 音声録音の設定
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
THRESHOLD = 800  # 音量のしきい値
SILENCE_DURATION = 2  # 録音終了と判断する無音の持続時間（秒）

def is_silent(data_chunk):
    """データチャンクが無音かどうかを判定する"""
    audio_data = np.frombuffer(data_chunk, dtype=np.int16)
    return np.max(audio_data) < THRESHOLD

class Voice2Text:
    def __init__(self):
        self.model = whisper.load_model("large")
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    def transcribe(self, audio_file):
        result = self.model.transcribe(audio_file, language="ja")
        #delte audio file
        os.remove(audio_file)
        return result["text"]
    
    def host_transribe(self):
        frames = []
        silence_count = 0
        recording = False

        while True:
            try:
                data = self.stream.read(CHUNK)
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
            except KeyboardInterrupt:
                break

        print("録音終了")

        wf = wave.open("myvoice.wav", 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        result = self.transcribe("myvoice.wav")
        return result["text"]
    


if __name__ == "__main__":
    model = Voice2Text()

    # 音声データをモデルに入力して文字起こし
    result = model.host_transribe()
    # 結果を表示
    print(result)