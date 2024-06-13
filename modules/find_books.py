from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import HumanMessagePromptTemplate
import base64
from dotenv import load_dotenv

load_dotenv()

SYS_MESSAGE = """あなたは物体検出と文字認識を正確に行う機械です.与えられた画像に写っている本のタイトルを左から順に以下のformatで列挙してください
    format: <STA>, 本の名前1, 本の名前2, 本の名前3, ..., 本の名前N, <END>

    タイトル以外の文字列は出力しないでください．
    もし本のタイトルが読み取れない場合は，その本の名前を<UNK>としてください．
    一冊も本が写っていない場合は，<NONE>のみを出力してください．
    それ以外の場合は，<ERR>のみを出力してください．
    上で述べたformat以外の出力は受け付けられません.
"""

class BookFinder:
    
    def __init__(self):
        self.model = ChatOpenAI(model="gpt-4o")


    def _process_image(self, image_path):
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return encoded_string
    
    def find_books(self, image_path):
        encoded_string = self._process_image(image_path)
        image_template = {
            "image_url": {"url": f"data:image/jpg;base64,{encoded_string}"},
        }
    
        human_template = HumanMessagePromptTemplate.from_template([image_template])
        prompt = ChatPromptTemplate.from_messages([("system", (SYS_MESSAGE)), human_template])

        chain = prompt | self.model

        response = chain.invoke({})
        return response.content
    

if __name__ == "__main__":
    bf = BookFinder()
    print(bf.find_books("./sample_image.jpg"))