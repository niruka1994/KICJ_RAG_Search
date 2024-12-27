import os
import streamlit as st
import tiktoken
import json
import uuid
import pickle

from dotenv import load_dotenv
from datetime import datetime
from langchain_core.runnables import RunnableConfig
from langchain_core.tracers import LangChainTracer
from langchain_core.tracers.run_collector import RunCollectorCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from langsmith import Client
from streamlit_feedback import streamlit_feedback
from operator import itemgetter
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever
from langchain_core.messages import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.document_transformers import LongContextReorder

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
LANGCHAIN_PROJECT = os.environ.get('LANGCHAIN_PROJECT')

st.set_page_config(
    page_title="KICJ RAG Search",
    page_icon="	:face_with_hand_over_mouth:")

st.title("_KICJ :red[RAG Search]_ 	:face_with_hand_over_mouth:")

year = ['ALL', 2024, 2023, 2022, 2021, 2020]
selected_year = st.selectbox('조회년도 선택', year)

session_id = uuid.uuid4()

client = Client()
ls_tracer = LangChainTracer(project_name=LANGCHAIN_PROJECT, client=client)
run_collector = RunCollectorCallbackHandler()
cfg = RunnableConfig()
cfg["callbacks"] = [ls_tracer, run_collector]
cfg["configurable"] = {"session_id": "any"}


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


def save_messages_to_txt(messages):
    # 현재 위치에서 Conversation 폴더 생성 또는 사용
    base_dir = "Conversation"
    os.makedirs(base_dir, exist_ok=True)

    # 현재 시간 형식으로 폴더 생성
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(base_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # 파일 경로 설정
    file_path = os.path.join(output_dir, "Conversation.txt")

    # 메시지 기록을 파일에 저장
    with open(file_path, 'w', encoding='utf-8') as file:
        for message in messages:
            role = message.role
            content = message.content
            file.write(f"{role}: {content}\n")

    return file_path


def get_session_history(session_ids: str) -> BaseChatMessageHistory:
    if session_ids not in st.session_state["store"]:
        st.session_state["store"][session_ids] = ChatMessageHistory()
    return st.session_state["store"][session_ids]


def load_bm25_retriever(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            bm25_retriever = pickle.load(f)
        return bm25_retriever
    else:
        return None


def get_vectorstore(year):
    # BM25Retriever 객체를 저장할 파일 경로 설정
    bm25_filepath = f"./BM25_cache/BM25_retriever_{year}.pkl"
    # 저장된 BM25Retriever 객체 불러오기
    bm25_retriever = load_bm25_retriever(bm25_filepath)
    # FAISS 객체를 저장할 폴더 경로 설정
    faiss_folder_path = "./Faiss_cache/"
    faiss_index_name = f"Faiss_index_{year}"

    # FAISS 객체를 로드 또는 생성

    store = LocalFileStore("./Embedding_cache/")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", disallowed_special=())
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings, store, namespace=embeddings.model
    )
    faiss_vector = FAISS.load_local(
        folder_path=faiss_folder_path,
        embeddings=cached_embeddings,
        index_name=faiss_index_name,
        allow_dangerous_deserialization=True
    )

    # FAISS 객체에서 retriever 생성
    faiss_retriever = faiss_vector.as_retriever(search_kwargs={"k": 2})

    # 앙상블 retriever 생성
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever], weight={0.7, 0.3}, search_type="mmr"
    )

    return ensemble_retriever


def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)


def print_messages():
    if "messages" in st.session_state and len(st.session_state["messages"]) > 0:
        for chat_message in st.session_state["messages"]:
            st.chat_message(chat_message.role).write(chat_message.content)


def reorder_documents(docs):
    reordering = LongContextReorder()
    reordered_docs = reordering.transform_documents(docs)
    return reordered_docs


# LCEL에서 내용통합 역할과 json파일로 떨구는 역할
def format_docs(docs):
    folder_path = os.path.join(os.getcwd(), 'Retrieved')
    os.makedirs(folder_path, exist_ok=True)

    file_name = f"{session_id}.json"
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump([doc.__dict__ for doc in docs], file, ensure_ascii=False, indent=4)
    return docs


# 떨궈진 json파일 로드
def load_meta_from_file(session_id):
    file_path = os.path.join(os.getcwd(), 'Retrieved', f"{session_id}.json")
    with open(file_path, 'r', encoding='utf-8') as file:
        meta_data = json.load(file)
    return meta_data


if "last_run" not in st.session_state:
    st.session_state["last_run"] = None

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "store" not in st.session_state:
    st.session_state["store"] = dict()

print_messages()

st.session_state.ensemble_retriever = get_vectorstore(selected_year)

if user_input := st.chat_input("메시지를 입력해주세요"):
    st.chat_message("user").write(f"{user_input}")
    st.session_state["messages"].append(ChatMessage(role="user", content=user_input))

    with st.chat_message("assistant"):
        with st.spinner("생성중.."):
            stream_handler = StreamHandler(st.empty())

            ensemble_retriever = st.session_state.get("ensemble_retriever")

            llm = ChatOpenAI(model_name='gpt-4o', temperature=0, streaming=True, callbacks=[stream_handler])
            query_llm = ChatOpenAI(model_name='gpt-4o-mini', temperature=0)

            multiquery_retriever = MultiQueryRetriever.from_llm(
                retriever=ensemble_retriever,
                llm=query_llm,
            )

            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """
                        You are a helpful assistant.
                        Answer questions using only the following context.
                        If you don't know the answer just say you don't know, don't make it up:
                        \n\n
                        "{context},
                        """
                    ),
                    MessagesPlaceholder(variable_name="history"),
                    ("human", "{question}"),
                ]

            )

            chain = (
                    {"context": itemgetter("question") | multiquery_retriever | reorder_documents | format_docs,
                     "question": itemgetter("question"), "history": itemgetter("history")}
                    | prompt
                    | llm
            )

            chain_with_memory = (
                RunnableWithMessageHistory(
                    chain,
                    get_session_history,
                    input_messages_key="question",
                    history_messages_key="history",
                )
            )

            response = chain_with_memory.invoke(
                {"question": user_input}, cfg
            )

            meta = load_meta_from_file(session_id)

            st.session_state["messages"].append(ChatMessage(role="assistant", content=response.content))
            st.session_state.last_run = run_collector.traced_runs[0].id
            save_messages_to_txt(st.session_state["messages"])

        with st.expander("참조된 문서"):
            for i in range(len(meta)):
                page_number = int(meta[i]['metadata']['page'])
                content = meta[i]['page_content']
                source = meta[i]['metadata']['source']

                markdown_content = f"***{source}***, ***{page_number}p***"
                st.markdown(markdown_content, unsafe_allow_html=True, help=f"{content}")

if st.session_state.get("last_run"):
    feedback = streamlit_feedback(
        feedback_type="faces",
        optional_text_label="[선택] 의견을 입력해주세요",
        key=f"feedback_{st.session_state.last_run}",
    )

    if feedback:
        scores = {"😀": 5, "🙂": 4, "😐": 3, "🙁": 2, "😞": 1}
        client.create_feedback(
            st.session_state.last_run,
            feedback["type"],
            score=scores[feedback["score"]],
            comment=feedback.get("text", None)
        )
        st.toast("피드백 저장완료", icon="🤲")
