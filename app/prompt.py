from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

WELCOME_MESSAGE = """\
Welcome to Introduction to LLM App Development Sample PDF QA Application!
To get started:
1. Upload a PDF or text file
2. Ask any question about the file!
"""

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Please act as an expert financial analyst when you answer the questions and pay special attention to the financial statements.  Operating margin is also known as op margin and is calculated by dividing operating income by revenue.

Given the following extracted parts of a long document and the conversation history, create a final answer with references ("SOURCES"). If you don't know the answer, just say that you don't know. Don't try to make up an answer.

ALWAYS return a "SOURCES" field in your answer, with the format "SOURCES: <source1>, <source2>, <source3>, ...".

Context from documents:
{context}"""
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ]
)
