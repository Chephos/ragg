import ollama
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models

logger = logging.getLogger(__name__)
# file_path = "nke-10k-2023.pdf"
# file_path = "https://icseindia.org/document/sample.pdf"


class Doc:

    @classmethod
    def document_loader(cls, file_path_or_url: str):
        print(f"Loading {file_path_or_url} for chunking")
        loader = PyPDFLoader(file_path_or_url)
        docs = loader.load()
        print(f"Document of successfully loaded: {len(docs)}")
        return docs

    @classmethod
    def split_docs_into_chunks(cls, docs, chunk_size=1000, chunk_overlap=200):
        print("Chunking document...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
        )
        all_splits = text_splitter.split_documents(docs)
        print(f"Document chunked into {len(all_splits)} bits")

        return all_splits


class Qdrant:
    client = QdrantClient("http://localhost:6333")
    MODEL_NAME = "BAAI/bge-small-en-v1.5"
    EMMBEDDING_MODEL = SentenceTransformer(MODEL_NAME)

    @classmethod
    def embed_chunks(cls, chunks):
        print("Embedding chunks...")
        unembedded_chunks = [{"content": data.page_content} for data in chunks]
        embeddings_sentences = cls.EMMBEDDING_MODEL.encode(
            [sentence.page_content for sentence in chunks]
        )
        print("Embedding successful")
        return embeddings_sentences, unembedded_chunks

    @classmethod
    def add_to_vector_db(cls, collection_name, embeddings, payload):
        print("Adding vectors to db...")
        if not cls.client.collection_exists(collection_name):
            cls.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=cls.client.get_embedding_size(cls.MODEL_NAME),
                    distance=models.Distance.COSINE,
                ),  # size and distance are model dependent
            )

        cls.client.upload_collection(
            collection_name=collection_name,
            wait=True,
            vectors=[v for v in embeddings],
            payload=payload,
        )
        print("Datastore prepared successfully")

    @classmethod
    def retrieve(cls, query: str, collection_name):

        query_vector = cls.EMMBEDDING_MODEL.encode(query)
        search_result = cls.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=5,
            with_payload=True,
        ).points
        return search_result

    @classmethod
    def prepare_data_store(cls, file_path_or_url, collection_name):
        docs = Doc.document_loader(file_path_or_url)
        chunks = Doc.split_docs_into_chunks(docs)
        embeddings, unembedded_text = cls.embed_chunks(chunks)
        cls.add_to_vector_db(collection_name, embeddings, unembedded_text)


class LLM:
    MODEL_NAME = "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF"

    @classmethod
    def generate_response(cls, search_result, input_query):

        instruction_prompt = f"""You are a helpful chatbot.
        Use only the following pieces of context to answer th question. Don't make up any new information:
        {'\n'.join([f" - {chunk.payload}" for chunk in search_result])}"""

        stream = ollama.chat(
            model=cls.MODEL_NAME,
            messages=[
                {"role": "system", "content": instruction_prompt},
                {"role": "user", "content": input_query},
            ],
            stream=True,
        )

        print("Chatbot response:")
        result = ""
        for chunk in stream:
            result += chunk["message"]["content"] + ""
            print(chunk["message"]["content"], end="", flush=True)

        return result


class Rag:
    @classmethod
    def search(cls, query):

        result = Qdrant.retrieve(query=query, collection_name="rag_test_collection")
        LLM.generate_response(search_result=result, input_query=query)


if __name__ == "__main__":

    Qdrant.prepare_data_store(
        file_path_or_url="https://arxiv.org/pdf/2408.09869",
        collection_name="rag_test_collection",
    )
    input_query = input("Ask your question: ")
    Rag.search(input_query)
