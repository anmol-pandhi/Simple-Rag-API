#imports
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv
import google.generativeai as genai
import chromadb
#loading file and chunking
def load_and_chunk(filepath:str, chunk_size: int = 500, overlap: int = 50) -> list:
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    
    return chunks

document_chunks = load_and_chunk("Sample.txt")
# print(f"Total chunks created: {len(document_chunks)}")
# print(f"First chunk preview: {document_chunks[0][:200]}")

#chromadb setup and collection
chroma_client=chromadb.Client()
collection= chroma_client.create_collection(name="document_chunks")
collection.add(
     documents=document_chunks,
    ids=[f"chunk_{i}" for i in range(len(document_chunks))]
)

print(f"Stored {collection.count()} chunks in ChromaDB")








load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)


model = genai.GenerativeModel("gemini-2.5-flash")


#routing
app = FastAPI(title="RAG Project API")

class Question(BaseModel):
    question: str

@app.post("/ask")
def ask_question(payload: Question):

    results= collection.query(
        query_texts=[payload.question],
        n_results=2
    )
    retrieved_context = results['documents'][0][0]
    prompt=f"""
You are a helpful assistant. Use the following context to answer the question without any symbols just plain text.
Context: {retrieved_context}
Question: {payload.question}"""
 
    response = model.generate_content(prompt)
    return {"answer": response.text , "\n context": retrieved_context}

@app.get("/")
def health_check():
    return {"status": "alive"}
#starting server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
