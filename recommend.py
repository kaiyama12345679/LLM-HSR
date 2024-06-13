from dotenv import load_dotenv
import sqlite3
import uuid
from logging import getLogger, StreamHandler
import logging
load_dotenv()
from modules.txt2voice import txt2voice
from modules.record_audio import Voice2Text, record_audio
from modules.recommender import Recommender


if __name__ == "__main__":
    books = ["容疑者Xの献身", "解析入門1", "月刊少女野崎くん", "ハリー・ポッターと賢者の石", "ゴールデンカムイ", "ロシア語でボソッとデレるアーリャさん"]
    txt2voice("ずんだもんなのだ．読んでみたい本の特徴を教えるのだ．")
    record_audio("myvoice.wav")
    v2t = Voice2Text()
    query = v2t.transcribe("myvoice.wav")
    txt2voice(f"わかったのだ．{query}の特徴を持つ本を探すのだ．")
    recommender = Recommender("books.db")

    top_books = recommender.get_recommendations(books, query, 1)


    txt2voice(f"わかったのだ．多分{top_books[0]}がおすすめなのだ．")

