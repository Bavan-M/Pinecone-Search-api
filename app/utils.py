import os
import openai
import wikipediaapi
from pinecone import Pinecone, ServerlessSpec
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from typing import List

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
wiki_wiki = wikipediaapi.Wikipedia(
    user_agent="PineconeSearchApp/1.0",
    language="en",
    extract_format=wikipediaapi.ExtractFormat.WIKI
)
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

def get_embedding(text: str) -> List[float]:
    response = openai.embeddings.create(
        input=text,
        model="text-embedding-3-small"  # or "text-embedding-ada-002"
    )
    return response.data[0].embedding

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
)

def fetch_wikipedia_documents(topic: str, max_pages: int = 5):  # Reduced from 100 to 5
    page = wiki_wiki.page(topic)
    if not page.exists():
        return []
    
    documents = []
    main_page_content = page.summary
    if main_page_content:
        main_chunks = text_splitter.split_text(main_page_content)  # Fixed variable name

        for i, chunk in enumerate(main_chunks):
            documents.append({
                "title": f"{page.title} - Part {i+1}",
                "content": chunk,
                "page_id": page.pageid,
                "chunk_id": i,
                "original_title": page.title,
                "is_main_page": True
            })
    
    for link in page.links.values():
        if len(documents) >= max_pages * 5:
            break
        if link.exists() and link.summary:
            link_chunks = text_splitter.split_text(link.summary)
            for i, chunk in enumerate(link_chunks):
                documents.append({
                    "title": f"{link.title} - Part {i+1}",
                    "content": chunk,
                    "page_id": link.pageid,
                    "chunk_id": i,
                    "original_title": link.title,
                    "is_main_page": False
                })
    return documents

def setup_pinecone_index(index_name: str = "wiki-search", dimension: int = 1536):
    """Fixed Pinecone index setup with proper error handling"""
    try:
        # List all indexes
        existing_indexes = pc.list_indexes()
        index_names = [index.name for index in existing_indexes.indexes] if hasattr(existing_indexes, 'indexes') else []
        
        if index_name not in index_names:
            print(f"Creating new index: {index_name}")
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            # Wait for index to be ready
            import time
            time.sleep(10)  # Wait 10 seconds for index to initialize
        else:
            print(f"Using existing index: {index_name}")
        
        return pc.Index(index_name)
    
    except Exception as e:
        print(f"Error in setup_pinecone_index: {e}")
        # Try to connect anyway
        try:
            return pc.Index(index_name)
        except Exception as final_error:
            raise Exception(f"Failed to connect to Pinecone index: {final_error}")