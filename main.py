from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_index.llms.openai import OpenAI
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.agent.workflow import AgentWorkflow
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
app = FastAPI()

# create RAG tool 
document = SimpleDirectoryReader("knowledge").load_data()
index = VectorStoreIndex.from_documents(document)
query_engine = index.as_query_engine()


async def search_document(query: str) -> str:
    response = await query_engine.aquery(query)
    return str(response)

agent = AgentWorkflow.from_tools_or_functions(
    [search_document],
    llm=OpenAI(model="gpt-3.5-turbo"),
    system_prompt= """You are a helpful assistent that answers questions based on provided CSV knowledge."""
)

class QueryReq(BaseModel):
    query:str

@app.post("/query")
async def query_chatbot(req: QueryReq):
    try:
        response = await agent.run(req.query)
        return{"response": response}
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)