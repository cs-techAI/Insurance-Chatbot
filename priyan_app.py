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

# database setup
DB_FILE = "token_usage.db"  # db file name is defined

def calculate_token_count(text, model="deepseek-chat"):  # cal no. of tokens in the text
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    return len(tokens)

# initialize chroma DB
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # embedding model
chroma_client = chromadb.PersistentClient(path="new_insurance")  # used for searching similarity text
collection = chroma_client.get_or_create_collection(name="insurance_embeddings")  # execute 2

# extract and display PDF content
def pdf_process(file):
    pdf_reader = PdfReader(file)  # extracts all the texts present in the pdf
    file_content = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
    st.text_area("📄 Extracted PDF Content", file_content[:2000])  # show first 2000 characters
    return file_content

# chunk text
def chunk_text(text, chunk_size=400, chunk_overlap=50):   # breaks down text into small chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap) 
    return text_splitter.split_text(text)   # done to store data efficiently / improves search accuracy 
    # this doesnt break text

# generate embeddings
def generate_embeddings(text):   # converts text into numerical vectors
    return embedding_model.encode(text.lower().strip()).tolist()  # used for semantic search in the chatbot

# store chunks in chroma DB
def store_chunks(chunks):   # takes in list of chunks
    stored_data = collection.get()  # retrieves all stored embeddings from chroma DB
    if stored_data.get("ids"):   
        # {
  #"ids": ["0", "1", "2"],
  #"embeddings": [[0.12, 0.32, ...], [0.43, 0.91, ...]],
  #"metadatas": [{"text": "First chunk"}, {"text": "Second chunk"}]
#} 
        collection.delete(ids=stored_data.get("ids"))  # delete old data

    st.write(f"✅ Cleared Old Data. Now storing {len(chunks)} new chunks...")

    for i, chunk in enumerate(chunks):  # loops through every chunks i - index and chunk - actual text chunk
        embedding = generate_embeddings(chunk)  # func call
        chunk_id = str(i)     # stores text embeddings in chroma DB so they can be searched
        collection.add(ids=[chunk_id], embeddings=[embedding], metadatas=[{"text": chunk}])

    st.success(f"✅ Stored {len(chunks)} Chunks in Vector DB")

# query stored embeddings
def query_embeddings(query_text, top_k=5):   # finds top 3 most relevant text from the stored db
    query_embedding = generate_embeddings(query_text)  # it retrieves insurance related info
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)  # compares query emb with stored emb
    return results

# call DeepSeek API for response 
def chatbot_with_deepseek(context, user_query, api_key):
    url = "https://api.deepseek.com/v1/chat/completions"  # request is sent to this url
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}  # informs the api that json data is sent
    full_prompt = f"""
You are an AI specializing in **insurance extraction**. Your task is to extract structured insurance details 
from the provided context. The user may upload documents related to various types of insurance, such as bike insurance, 
car insurance, health insurance, and more. Ensure that the extracted details are relevant and formatted properly.

If any detail is missing, **infer it based on context instead of leaving it blank**.

**Extract the following details if available:**
- **Owner Name**
- **Insurance Type** (Bike, Car, Truck, Bus etc.)
- **Vehicle Model** (if applicable)
- **Registration Number** (if applicable)
- **Insurance Provider**
- **Plan Name** (if applicable)
- **Insured Value**
- **Policy Expiry Date**
- **Additional Coverage Details**

**Context:**
{context}

**User Question:** {user_query}

**Provide a confidence score (0-100%) based on accuracy.**
"""
    tokens_used = calculate_token_count(full_prompt)  # cal no. of tokens 
    # defines the request body that will be sent to the DeepSeek API
    payload = {"model": "deepseek-chat", "messages": [{"role": "user", "content": full_prompt}]}  # message format is similar openai

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))  # sends post request to deepseek also converts payload dict. into json
        if response.status_code != 200:  # executes if the response code is not 200
            return "API Error", None

        # api response structure "choices", "message", "content"
        response_json = response.json()   # Converts the API response into JSON format
        response_content = response_json.get("choices", [{}])[0].get("message", {}).get("content", "No response")
        # content contains the extracted insurance details

        # extract confidence score
        confidence_score = None  # if none still the code runs and dont break
        for line in response_content.split("\n"):  # scans each line of the AI response to find the confidence score
            if "Confidence Score" in line or "Accuracy Score" in line:
                confidence_score = line.split(":")[-1].strip().replace("%", "")

        if confidence_score and confidence_score.isdigit():
            confidence_score = int(confidence_score)   # converts to int only if ai responds in a number
        else:
            confidence_score = "Not provided"

    except (json.JSONDecodeError, requests.RequestException):
        return "Request failed.", None  # catches any errors like API timeout, bad response format

    log_data_to_arctic("deepseek_chatbot", full_prompt, response_content, 0, tokens_used)

    return response_content, confidence_score

# streamlit UI
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
    results = query_embeddings(user_query)  # searches for relevant text  # func call
    context = "\n".join([meta["text"] for meta in results.get("metadatas", [{}])[0]])  # stores actual texts in context
    response, confidence_score = chatbot_with_deepseek(context, user_query, deepseek_api_key)

    st.text_area("🧠 AI Response:", value=response, height=200)


    # convert AI response to CSV
    response_data = [line.split(": ", 1) for line in response.split("\n") if ": " in line]  
    # as per the prompt there will be : so i used it to split 
    df = pd.DataFrame(response_data, columns=["Column 1", "Column 2"])  # converts into panda df

    csv_buffer = StringIO()  # creates csv file in memory
    df.to_csv(csv_buffer, index=False)  # converts df to CSV

    st.download_button("📥 Download Insurance Details CSV", csv_buffer.getvalue(), "insurance_details.csv", "text/csv")
