from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import HumanMessagePromptTemplate
import base64
from dotenv import load_dotenv
import difflib
import cv2

load_dotenv()

SYS_MESSAGE = """あなたは物体検出と文字認識を正確に行う機械です.与えられた画像に写っている本の数とそのタイトルを左から順に以下のformatで列挙してください

    format: N<SEP>本のタイトル1<SEP>本のタイトル2<SEP>本のタイトル3<SEP>...<SEP>本のタイトルN

    本のタイトル以外の文字は絶対に含めないでください．
    もし本のタイトルが読み取れない場合は，その本の名前を<UNK>としてください．
    一冊も本が写っていない場合は，<NONE>のみを出力してください．
    それ以外の場合は，<ERR>のみを出力してください．
    上で述べたformat以外の出力は受け付けられません.
"""

SYS_MESSAGE = """あなたは物体検出と文字認識を正確に行う機械です.画像に写っている，最も近い本の表紙に記載されているその本のタイトルのみを出力してください．
本の表紙に人の手など他の物体が被っていても続行してください．
可能な限り，本のタイトル名を出力することを求めますが，どうしても読み取れない場合は，<UNK> のみを出力してください
そして，あなたの応答に，本のタイトル以外の文字は絶対に含めないでください.
"""

class BookFinder:
    
    def __init__(self, titles):
        self.model = ChatOpenAI(model="gpt-4o")
        self.titles = titles

    def _process_image_from_raw(self, image):
        _, buffer = cv2.imencode('.png', image)

        encoded_string = base64.b64encode(buffer).decode("utf-8")

        return encoded_string

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
        book_number, book_names = self.parse_book_name(response.content)
        book_names = self.recover_name(book_names)
        return book_number, book_names
    
    def find_books_from_raw(self, image):
        encoded_string = self._process_image_from_raw(image)
        image_template = {
            "image_url": {"url": f"data:image/jpg;base64,{encoded_string}"},
        }
    
        human_template = HumanMessagePromptTemplate.from_template([image_template])
        prompt = ChatPromptTemplate.from_messages([("system", (SYS_MESSAGE)), human_template])

        chain = prompt | self.model

        response = chain.invoke({})
        if response.content != "<UNK>":
            correct_name = difflib.get_close_matches(response.content, self.titles, n=1, cutoff=0)[0]
            return correct_name
        else:
            return response.content
        # book_number, book_names = self.parse_book_name(response.content)
        # book_names = self.recover_name(book_names)
        # return book_number, book_names
    
    @staticmethod
    def parse_book_name(detected_books: str):
        if detected_books == "<NONE>":
            return 0, []
        elif detected_books == "<ERR>":
            return -1, []
        else:
            book_number = detected_books.split("<SEP>")[0]
            books = detected_books.split("<SEP>")[1:]
            return book_number, [book for book in books]
        
    def recover_name(self, incomplete_titles):
        if len(incomplete_titles) < 1:
            return []
        results = [difflib.get_close_matches(incomplete_title, self.titles, n=1, cutoff=0)[0] for incomplete_title in incomplete_titles]
        return results
    

if __name__ == "__main__":
    bf = BookFinder()
    book_number, book_names = bf.find_books("test.jpg")
    print(book_number, book_names)