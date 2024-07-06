from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import HumanMessagePromptTemplate
import base64
from dotenv import load_dotenv

load_dotenv()


SYS_MESSAGE = """あなたはユーザーと会話できるチャットボットです．基本的には，ユーザーの話した内容に対して会話を行ってください．
ただし，ユーザーが「本を読みたい」もしくは，「本を選んでほしい」や「おすすめの本を教えて」といった趣旨の発言をした場合は通常の応答をせず， <REQ> のみを返してください．この場合，<REQ> 以外の返答は行わないでください．"""

class Interaction:
    def __init__(self):
        self.chatbot = ChatOpenAI(model="gpt-4o")
        print("Chatbot is ready!")

    def interact(self, query: str):
        message = [
                (
                    "system",
                    SYS_MESSAGE
                ),
                (
                    "human",
                    query
                )
            ]

        response = self.chatbot.invoke(message)
        return response.content