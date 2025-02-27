import streamlit as st  # for UI
import json   # for API requests and response formatting
import requests  # to send API calls
import tiktoken # for tokens
import chromadb
import pandas as pd
from io import StringIO  # for csv data 
from pypdf import PdfReader  # to read pdf
from sentence_transformers import SentenceTransformer  # embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter  # splitting texts
from db_utils import create_table, log_data_to_arctic   # execute 1



# Database setup
DB_FILE = "token_usage.db"  # db file name is defined

def calculate_token_count(text, model="deepseek-chat"):  # cal no. of tokens in the text
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    return len(tokens)


# Initialize ChromaDB
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # embedding model
chroma_client = chromadb.PersistentClient(path="new_insurance")  # used for searching similarity text
collection = chroma_client.get_or_create_collection(name="insurance_embeddings")  # execute 2

# Extract and display PDF content
def pdf_process(file):
    pdf_reader = PdfReader(file)  # extracts all the texts present in the pdf
    file_content = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
    st.text_area("📄 Extracted PDF Content", file_content[:2000])  # show first 2000 characters
    return file_content



# Chunk text
def chunk_text(text, chunk_size=400, chunk_overlap=50):  # breaks down text into small chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(text)  # improves search accuracy 


# Generate embeddings
def generate_embeddings(text):  # converts text into numerical vectors
    return embedding_model.encode(text.lower().strip()).tolist()  # used for semantic search in the chatbot



# Store chunks in ChromaDB
def store_chunks(chunks):  # takes in list of chunks
    stored_data = collection.get()  # retrieves all stored embeddings from ChromaDB
    if stored_data.get("ids"):   
        collection.delete(ids=stored_data.get("ids"))  # delete old data

    print(f"✅ Cleared Old Data. Now storing {len(chunks)} new chunks...")  # Console Output


    for i, chunk in enumerate(chunks):  # loops through every chunk
        embedding = generate_embeddings(chunk)
        chunk_id = str(i)  # stores text embeddings in ChromaDB for searching
        collection.add(ids=[chunk_id], embeddings=[embedding], metadatas=[{"text": chunk}])

    st.success(f"✅ Stored {len(chunks)} Chunks in Vector DB")



# Query stored embeddings
def query_embeddings(query_text, top_k=5):   # finds top 3 most relevant text from the stored db
    query_embedding = generate_embeddings(query_text)  # it retrieves insurance related info
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)  # compares query emb with stored emb

    # Debugging: Print full results to terminal instead of UI
    print("🔍 **Query Results:**", results)


    # Extract text from metadata
    retrieved_texts = []
    metadata_list = results.get("metadatas", [])

    if metadata_list and isinstance(metadata_list, list):  
        for metadata in metadata_list:  
            if isinstance(metadata, list):  # Flatten nested lists
                for meta in metadata:
                    if isinstance(meta, dict) and "text" in meta:  
                        retrieved_texts.append(meta["text"])
            elif isinstance(metadata, dict) and "text" in metadata:
                retrieved_texts.append(metadata["text"])

    if not retrieved_texts:  
        return ["No relevant data found in the database."]  

    return retrieved_texts  




# Call DeepSeek API for response
def chatbot_with_deepseek(context, user_query, api_key):  
    url = "https://api.deepseek.com/v1/chat/completions"  # request is sent to this url
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}  # informs the api that json data is sent

    # Debugging: Print extracted context to terminal instead of UI
    print("📌 **Final Context Sent to LLM:**", context)


    full_prompt = f"""
You are an AI specializing in **insurance extraction**. Your task is to extract structured insurance details 
from the provided context. The user may upload documents related to various types of insurance, such as 
bike insurance, car insurance, health insurance, and more. 
Ensure that the extracted details are relevant and **ONLY present in the provided context**.

If a detail is **not found in the context**, **DO NOT** generate a response for it.

**Context:**
{context}

**User Question:** {user_query}

**Provide a confidence score (0-100%) based on accuracy.**
"""

    tokens_used = calculate_token_count(full_prompt)  
    payload = {"model": "deepseek-chat", "messages": [{"role": "user", "content": full_prompt}]}  

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))  
        if response.status_code != 200:  
            return "API Error", None

        response_json = response.json()  
        response_content = response_json.get("choices", [{}])[0].get("message", {}).get("content", "No response")


        # Extract confidence score
        confidence_score = None  
        for line in response_content.split("\n"):  
            if "Confidence Score" in line or "Accuracy Score" in line:
                confidence_score = line.split(":")[-1].strip().replace("%", "")

        if confidence_score and confidence_score.isdigit():
            confidence_score = int(confidence_score)   
        else:
            confidence_score = "Not provided"

    except (json.JSONDecodeError, requests.RequestException):
        return "Request failed.", None  



    log_data_to_arctic("deepseek_chatbot", full_prompt, response_content, 0, tokens_used)

    return response_content, confidence_score



# Streamlit UI
st.set_page_config(page_title="Insurance AI", layout="wide")  # browser tab name
st.title("📄 Insurance AI ")  # execute 4

file = st.file_uploader("Upload an Insurance PDF", type=["pdf"])  # execute 5
if file:         # execute 6
    st.success("✅ File uploaded successfully!")
    pdf_text = pdf_process(file)  # func call
    chunks = chunk_text(pdf_text) # func call
    st.markdown(f"📊 **Total Chunks:** {len(chunks)}")  # shows total chunks # output
    
    if st.button("🔄 Convert to Embeddings"):   # execute 7
        store_chunks(chunks)  # func call

user_query = st.text_input("💬 Ask about insurance:")  # execute 8
deepseek_api_key = st.text_input("🔑 Enter DeepSeek API Key:", type="password")   # execute 9


if deepseek_api_key and user_query:   # execute 10
    retrieved_texts = query_embeddings(user_query)  
    context = "\n".join(retrieved_texts)  

    # 🔍 Debugging: Ensure context is not empty
    if not context.strip():
        st.warning("⚠️ No relevant data was found in the database. AI will not generate a response.")

    response, confidence_score = chatbot_with_deepseek(context, user_query, deepseek_api_key)

    st.text_area("🧠 AI Response:", value=response, height=200)


    # ✅ **Fixed CSV Generation**
    response_lines = response.split("\n")  # Split response into lines
    response_data = []

    for line in response_lines:
        if ": " in line:  # Ensure line has a separator
            parts = line.split(": ", 1)
            if len(parts) == 2:
                response_data.append(parts)

    df = pd.DataFrame(response_data, columns=["Field", "Value"])  

    csv_buffer = StringIO()  
    df.to_csv(csv_buffer, index=False)  
    

    st.download_button("📥 Download Insurance Details CSV", csv_buffer.getvalue(), "insurance_details.csv", "text/csv")
