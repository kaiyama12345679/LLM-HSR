import simpleaudio
import json
import requests
import time
import os

def txt2voice(text, spekaer: int = 3):
    ut = time.time()

    #クエリ生成
    res1 = requests.post("http://localhost:50021/audio_query",
                        params={"text": text, "speaker": spekaer})
    # synthesis (音声合成するAPI)
    res2 = requests.post("http://localhost:50021/synthesis",
                        params={"speaker": spekaer},
                        data=json.dumps(res1.json()))
    # wavファイルに書き込み
    audio_file = f"{ut}.wav"
    with open(audio_file, mode="wb") as f:
        f.write(res2.content)

    playWav(audio_file)
    os.remove(audio_file)

def playWav(file):
    with open(file,"rb") as f:
        # wavファイル再生
        wav_obj = simpleaudio.WaveObject.from_wave_file(f)
        play_obj = wav_obj.play()
        play_obj.wait_done()


if __name__ == "__main__":
    txt2voice("滲み出す混濁の紋章、不遜なる狂気の器、湧き上がり・否定し・痺れ・瞬き・眠りを妨げる爬行(はこう)する鉄の王女、絶えず自壊する泥の人形、結合せよ、反発せよ、地に満ち 己の無力を知れ．破道の九十・黒棺")
