from record_audio import Voice2Text
from txt2voice import txt2voice
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import HumanMessagePromptTemplate
from recommender import Recommender
import base64
from dotenv import load_dotenv

load_dotenv()


SYS_MESSAGE = """あなたはユーザーと会話できるチャットボットです．基本的には，ユーザーの話した内容に対して会話を行ってください．ただし，語尾は「のだ」で終わるようにしてください．
ただし，ユーザーが「本を読みたい」もしくは，「本を選んでほしい」といった趣旨の発言をした場合に限り， <REQ> のみを返してください．この場合，<REQ> 以外の返答は行わないでください．"""




books = ["ハリーポッターと賢者の石", "ソードアートオンライン", "四月は君の嘘", "解析入門", "ゼロから作るDeep Learning"]

class Interaction:
    def __init__(self):
        self.voice2text = Voice2Text()
        self.chatbot = ChatOpenAI(model="gpt-4o")
        self.recomender = Recommender("books.db", verbose=True)
        print("Chatbot is ready!")
        txt2voice("やあ，僕はずんだもん．あなたの相手をするのだ．")

    def process(self):
        while True:
            text = self.voice2text.host_transribe()

            message = [
                (
                    "system",
                    SYS_MESSAGE
                ),
                (
                    "human",
                    text
                )
            ]

            response = self.chatbot.invoke(message)
            if response.content == "<REQ>":
                txt2voice("本を探してあげるのだ．どんな本が読みたいのだ？")
                query = self.voice2text.host_transribe()
                txt2voice("本を選んであげるのだ．ちょっとじかんがかかるのだ．")
                book = self.recomender.get_recommendations(books, query)
                txt2voice(f"本を選んだのだ．たぶんだけど, {book}がおすすめなのだ")
            else:
                txt2voice(response.content)


if __name__ == "__main__":
    interaction = Interaction()
    interaction.process()