from fastapi import FastAPI,HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
from typing import List

from .models import SearchQuery,SearchResult,TopicRequest
from .utils import get_embedding,fetch_wikipedia_documents,setup_pinecone_index

app=FastAPI(title="Pinecone Semantic Search API")
app.mount("/static",StaticFiles(directory="static"),name="static")

index=setup_pinecone_index()

PREDEFINED_TOPICS = [
    "Artificial Intelligence",
    "Machine Learning",
    "Deep Learning",
    "Natural Language Processing",
    "Computer Vision",
    "Data Science",
    "Neural Networks",
    "Robotics",
    "Quantum Computing",
    "Blockchain",
    "Internet of Things",
    "Cloud Computing",
    "Cybersecurity",
    "Virtual Reality",
    "Augmented Reality",
    "Big Data",
    "Python Programming",
    "JavaScript",
    "Web Development",
    "Mobile Applications"
]

@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

@app.get("/topics")
async def get_topic():
    return {"topics":PREDEFINED_TOPICS}

@app.post("/index-documents")
async def index_documents(topic_request:TopicRequest):
    try:
        topic=topic_request.topic
        if topic not in PREDEFINED_TOPICS:
            raise HTTPException(status_code=400,detail="Invalid Topic Selected")
        
        documents=fetch_wikipedia_documents(topic=topic)
        if not documents:
            raise HTTPException(status_code=400,detail=f"No document found for the topic {topic}")

        vectors=[]
        for doc in documents:
            embedding=get_embedding(doc['content'])
            vectors.append({
                "id":f"{doc['page_id']}_{doc['chunk_id']}",
                "values":embedding,
                "metadata":{
                    "title":doc['title'],
                    "content":doc['content'],
                    "page_id":doc['page_id'],
                    "chunk_id":doc['chunk_id'],
                    "is_main_page":doc.get("is_main_page",False),
                    "topic":topic
                }
            })

            batch_size=100
            for i in range(0,len(documents),batch_size):
                batch=vectors[i:i+batch_size]
                index.upsert(vectors)

            return {
                "message":f"Sucessfully indexed {len(documents)} documnent chunks for {topic}",
                "topic":topic,
                "document_count":len(documents)
            }
    except Exception as e:
        raise HTTPException(status_code=500,detail=f"Error ocuured while indexing {str(e)}")

@app.post("/search",response_model=List[SearchResult])
async def search_document(search_query:SearchQuery):
    try:
        query_embedding=get_embedding(search_query.query)
        results=index.query(vector=query_embedding,top_k=search_query.top_k,include_metadata=True,include_values=False)
        search_result=[]
        for match in results.matches:
            search_result.append(SearchResult(title=match.metadata['title'],content=match.metadata['content'],score=match.score,page_id=match.metadata['page_id']))
        return search_result
    except Exception as e:
        raise HTTPException(status_code=500,detail=f"Search Error : {str(e)}")

@app.delete("/clear-index")
async def clear_index():
    try:
        index.delete(delete_all=True)
        return {"message":"Index xleared Sucessfully"}
    except Exception as e:
        raise HTTPException(status_code=500,detail=f"Error Clearing Index {str(e)}")
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
