from fastapi import FastAPI

from services import Qdrant, Rag

app = FastAPI()


@app.get("/api/search")
def search_startup(q: str):
    # print("Loading Datastore...")
    Qdrant.prepare_data_store(
        file_path_or_url="https://arxiv.org/pdf/2408.09869",
        collection_name="rag_test_collection",
    )
    return {"result": Rag.search(q)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=8000)
