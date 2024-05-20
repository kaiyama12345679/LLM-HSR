from langchain_openai import ChatOpenAI
from langchain.agents import load_tools, create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
import torch
from dotenv import load_dotenv
load_dotenv()

template = '''Answer the following questions as best you can. You have access to the following tools:

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

prompt = PromptTemplate.from_template(template)

tools = ["google-search"]
llm = ChatOpenAI(model="gpt-3.5-turbo")
tools = load_tools(tools, llm=llm)
agent = create_react_agent(llm, tools, prompt)
agent_executor =AgentExecutor(agent=agent, tools=tools)

def get_content(title):
    book_template = """
    Please Describe the content of the book "{title}. You should provides as much specific information as possible about what the book covers, important, themes, genre, proper nouns, and a summary of the content".
    Do not include any information unrelated to the book's content.
    """
    book_prompt = PromptTemplate(template=book_template, input_variables={"title"})
    book_prompt = book_prompt.format(title=title)
    return agent_executor.invoke({"input": book_prompt})

def get_books_content(books):
    tasks = []
    for title in books:
        tasks.append(get_content(title))
    return tasks

if __name__ == "__main__":
    books = ["容疑者Xの献身", "解析入門1", "月刊少女野崎くん", "ハリー・ポッターと賢者の石", "ゴールデンカムイ"]
    user_preference = input("読みたい本についての要求：")
    if type(user_preference) != str:
        raise ValueError("Please enter a valid string")

    results = get_books_content(books)
    with open("results.txt", "w") as f:
        for result in results:
            f.write(result["output"] + "\n")
    embeddings = OpenAIEmbeddings()
    query_result = embeddings.embed_documents([result["output"] for result in results])

    book_data = torch.tensor(query_result)
    message = [
    ("system", "You are a dictionary that provides information about genres and contents of books which are requested by users. Do not include any specific titles of books in your response. "),
    ("human", user_preference)]
    preferences = llm.invoke(message)
    preference_emb = embeddings.embed_query(preferences.content)

    pfr = torch.tensor(preference_emb)
    pfr = pfr.repeat(book_data.shape[0], 1)
    similarity = torch.nn.functional.cosine_similarity(book_data, pfr, dim=1)

    max_index = torch.argmax(similarity)
    print(books[max_index.item()])

