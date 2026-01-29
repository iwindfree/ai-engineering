from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
from langchain_core.documents import Document

from dotenv import load_dotenv

load_dotenv(override=True)

MODEL="gpt-4.1-nano"
DB_NAME=str(Path(__file__).parent.parent / "vector_db")

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

def fetch_context(question: str) -> list[Document]:
    """주어진 질문에 대해 관련 컨텍스트 문서를 검색합니다.

    벡터 스토어에서 질문과 의미적으로 유사한 문서들을 검색하여 반환합니다.
    검색되는 문서 개수는 RETRIEVAL_K 상수에 의해 결정됩니다.

    Args:
        question: 사용자가 입력한 질문 문자열

    Returns:
        검색된 Document 객체들의 리스트
    """
    return retriever.invoke(question, k=RETRIEVAL_K)

def combined_question(question: str, history: list[dict] = []) -> str:
    """현재 질문과 대화 히스토리를 결합하여 하나의 질문 문자열로 만듭니다.

    대화 히스토리에서 사용자(user) 역할의 메시지들만 추출하여
    현재 질문 앞에 붙여 컨텍스트 검색에 활용할 수 있도록 합니다.
    이를 통해 이전 대화 맥락을 고려한 문서 검색이 가능해집니다.

    Args:
        question: 현재 사용자가 입력한 질문
        history: 이전 대화 히스토리 (role, content 키를 가진 딕셔너리 리스트)

    Returns:
        이전 질문들과 현재 질문이 결합된 문자열
    """
    prev_question = "\n".join(turn["content"] for turn in history if turn["role"] == "user")
    return prev_question + "\n" + question     

def answer_question(question: str, history: list[dict] = []) -> str:
    """RAG 방식으로 질문에 대한 답변을 생성합니다.

    1. 대화 히스토리와 현재 질문을 결합하여 검색 쿼리 생성
    2. 벡터 스토어에서 관련 컨텍스트 문서 검색
    3. 검색된 컨텍스트와 대화 히스토리를 포함한 프롬프트로 LLM 호출
    4. 생성된 답변 반환

    Args:
        question: 사용자가 입력한 현재 질문
        history: 이전 대화 히스토리 (role, content 키를 가진 딕셔너리 리스트)

    Returns:
        LLM이 생성한 답변 문자열
    """
    combined_q = combined_question(question, history)
    context_docs = fetch_context(combined_q)
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
    response = llm.invoke(messages)

    return response.content, context_docs