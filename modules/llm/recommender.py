from langchain_openai import ChatOpenAI
from langchain.agents import load_tools, create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
import torch
from dotenv import load_dotenv
import sqlite3
import uuid
from logging import getLogger, StreamHandler
import logging
from typing import List
import difflib
load_dotenv()

TEMPLATE = '''Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}'''

BOOK_TEMPLATE = """
Please Describe the content of the book "{title}. You should provides as much specific information as possible about what the book covers, important, themes, genre, characteristics and personalities of characters, and a summary of the content".
Do not include any sentences nor words unrelated to the book's content in your response.
"""


class Recommender():
    def __init__(self, db_path, books, verbose = False):
        self.logger = getLogger(__name__)
        self.stream_handler = StreamHandler()
        self.logger.addHandler(self.stream_handler)

        self.llm = ChatOpenAI(model="gpt-3.5-turbo")
        self.prompt = PromptTemplate.from_template(TEMPLATE)
        self.tools = load_tools(["google-search"], llm=self.llm)
        self.agent = create_react_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=verbose, handle_parse_error=True)

        self.embeddings = OpenAIEmbeddings()

        self.db = sqlite3.connect(db_path)
        self.cursor = self.db.cursor()

        self.cursor.execute("CREATE TABLE IF NOT EXISTS books (id TEXT PRIMARY KEY, title TEXT, content TEXT)")
        if books is not None:
            self.insert_books(books)
        self.db.commit()

    def get_books_content(self, books):
        tasks = []
        for title in books:
            tasks.append(self.get_content(title))
        return tasks

    def get_content(self, title):
        try:
            self.cursor.execute("SELECT content FROM books WHERE title=?", (title,))
            content = self.cursor.fetchone()
            assert content[0] is not None or content[0] != ""
            return {"title": title, "output": content[0]}
        except Exception as e:
            self.db.rollback()
            raise e
        
    def insert_books(self, titles):
        for title in titles:
            if not self.check_book_exists(title):
                try:
                    book_prompt = PromptTemplate(template=BOOK_TEMPLATE, input_variables={"title"})
                    book_prompt = book_prompt.format(title=title)
                    result = self.agent_executor.invoke({"input": book_prompt})
                    assert result["output"] is not None or result["output"] != ""
                    self.cursor.execute("INSERT INTO books (id, title, content) VALUES (?, ?, ?)", (str(uuid.uuid4()), title, result["output"]))
                except Exception as e:
                    self.db.rollback()
                    raise e
        self.db.commit()
        
    def check_book_exists(self, title):
        self.cursor.execute("SELECT content FROM books WHERE title=?", (title,))
        return self.cursor.fetchone() is not None
    
    def decode_correct_title(self, incomplete_titles):
        assert len(incomplete_titles) > 0
        self.cursor.execute("SELECT title FROM books")
        titles = [output[0] for output in self.cursor.fetchall()]
        results = [difflib.get_close_matches(incomplete_title, titles, n=1, cutoff=0)[0] for incomplete_title in incomplete_titles]
        return results
    
    def get_recommendations(self, query):
        self.cursor.execute("SELECT title FROM books")
        titles = [output[0] for output in self.cursor.fetchall()]
        idx2title = {idx: title for idx, title in enumerate(titles)}
        tasks = self.get_books_content(titles)
        embeddings = torch.tensor(self.embeddings.embed_documents([task["output"] for task in tasks]))
        
        message = [
            ("system", "You are a dictionary that provides information about genres and contents of books which are requested by users. Do not include any specific titles of books in your response. "),
            ("human", query)]
        pfr = self.llm.invoke(message)
        pfr_embedding = torch.tensor(self.embeddings.embed_query(pfr.content))
        pfr_embedding = pfr_embedding.unsqueeze(0).expand(embeddings.shape[0], -1)
        similarities = torch.nn.functional.cosine_similarity(embeddings, pfr_embedding, dim=1)

        top_idx = similarities.argsort(descending=True)[0].item()
        return idx2title[top_idx]

if __name__ == "__main__":
    recommender = Recommender("books.db", books=None, verbose=True)
    books = ["ハリーポッターと賢者の石", "ソードアートオンライン", "四月は君の嘘", "解析入門", "ゼロから作るDeep Learning"]
    recommender.insert_books(books)
    query = "ファンタジー系じゃない，でもラブコメの本を探しているんだけど"
    recommendded_book = recommender.get_recommendations(query)
    print(recommendded_book)