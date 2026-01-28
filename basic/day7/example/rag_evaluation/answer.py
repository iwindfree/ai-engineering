from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import  HumanMessage, SystemMessage, convert_to_messages
from langchain_core.schema import Document
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from dotenv import load_dotenv

load_dotenv(override=True)

MODEL="gpt-4.1-nano"
DB_NAME=str(Path(__file__).parent / "vector_db")

# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
RETRIEVAL_K = 10


SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
If relevant, use the given context to answer any question.
If you don't know the answer, say so.
Context:
{context}
"""

vectorstore = Chroma(
    persist_directory=DB_NAME,
    embedding_function=embeddings)

retriever = vectorstore.as_retriever()
llm = ChatOpenAI(model=MODEL, temperature=0)

def fecth_context(question: str) -> List[Document]:
    """Fetch relevant context documents for the given question."""
    return retriever.invoke(question, k=RETRIEVAL_K)

def combined_question(question: str, history: list[dict] = []) -> str:
    """Combine the question with chat history if available."""
    prev_question = "\n".join(turn["content"] for turn in history if turn["role"] == "user")
    return prev_question + "\n" + question     

def answer_question(question: str, history: list[dict] = []) -> str:
    """Answer the question using retrieved context and chat history."""
    combined_q = combined_question(question, history)
    context_docs = fecth_context(combined_q)
    context_text = "\n\n".join(doc.page_content for doc in context_docs)

    system_message = SystemMessage(content=SYSTEM_PROMPT.format(context=context_text))
    
    """
    System: (컨텍스트 + 지침)
    User: (히스토리를 포함한 하나의 질문)
    """
    #user_message = HumanMessage(content=combined_q)
    #messages = [system_message, user_message]
    #response = llm.invoke(messages=messages)
    """
    System: (컨텍스트 + 지침)
    User: ...
    Assistant: ...
    User: ...
    Assistant: ...
    User: (현재 질문)
    """
    messages = [system_message]
    messages.extend(convert_to_messages(history))
    messages.append(HumanMessage(content=question))
    response = llm.invoke(messages=messages)

    return response.content