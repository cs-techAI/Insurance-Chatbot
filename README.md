# 📌 Insurance AI Agent

## 🚀 Overview
This project is a **Streamlit-based application** that allows users to:
- **Upload a PDF file**
- **Extract and store text as vector embeddings** using **ChromaDB**
- **Retrieve relevant text chunks** based on user queries
- **Analyze insurance details** using **DeepSeek API**
- **Download extracted insurance data as CSV**

---

## ✨ Features
✅ **Upload PDF**: Accepts **.pdf** files for text extraction.  
✅ **Convert to Vector Embeddings**: Uses **MiniLM Sentence Transformer** for vectorization.  
✅ **View Stored Chunks**: Displays stored text chunks retrieved from **ChromaDB**.  
✅ **Chat with the Document**: Retrieves relevant content from stored chunks and interacts with **DeepSeek API**.  
✅ **Confidence Score (%)**: AI provides an accuracy score for extracted data.  
✅ **Download as CSV**: Get structured insurance details in a CSV format.  
✅ **Token Calculation**: Displays **token count** before sending API queries.  
✅ **Logging**: Tracks **API usage**, **response time**, and **token count** in **SQLite database**.  

---

## 📦 Installation
### **Prerequisites**
Ensure you have **Python 3.8+** installed on your system.

### **Install Dependencies**
Run the following command to install the required Python packages:
```sh
pip install streamlit chromadb sentence-transformers pypdf requests tiktoken sqlite3 langchain langchain_community
```

---

## ▶️ Usage
### **1. Run the Streamlit App**
```sh
streamlit run app.py
```

### **2. Upload a PDF File**
- Click the **Upload File** tab.
- Select a **PDF file**.
- The file will be processed automatically.

### **3. Convert to Vector Embeddings**
- Navigate to the **Convert to Vector Embeddings** tab.
- Click **Initialize Vectorization** to extract text.
- Click **Store Chunks in Vector DB** to store vectorized text.

### **4. View PDF Content**
- Go to the **View Data** tab.
- View the extracted text from the uploaded file.

### **5. View Stored Chunks**
- Navigate to the **View Stored Chunks** tab.
- Click **View Stored Chunks** to see stored text chunks and metadata.

### **6. Chat with the PDF**
- Navigate to the **ChatBot** tab.
- Enter your **question**.
- Enter your **DeepSeek API Key**.
- Click **Get Response** to receive an AI-generated answer.
- View **retrieved source chunks** for transparency.

### **7. Download Extracted Insurance Details**
- Click the **Download CSV** button to save structured insurance details.

---

## 🛠️ Project Structure
```
📂 Insurance-AI-Agent
│── app.py                 # Main Streamlit application
│── db_utils.py            # Handles logging and token calculation
│── requirements.txt       # Required dependencies
│── logs/                  # Stores API logs
│── README.md              # Project documentation
```

---

## ⚙️ Configuration
### **API Keys**
- **DeepSeek API Key** is required for chatbot functionality.
- The key must be entered in the **ChatBot tab** before making queries.

### **Token Limit**
- **Token count is displayed before sending queries**.
- Users should monitor token usage to avoid exceeding API limits.

---

## 📌 Example CSV Output
| **Column 1**                           | **Column 2** |
|----------------------------------------|------------------------------------|
| **Owner Name**                         | Rajesh Sharma                      |
| **Car Model**                          | Maruti Suzuki Swift 2022           |
| **Car Registration Number**            | MH-12-AB-5678                      |
| **Insured Value**                      | INR 8,00,000                       |
| **Insurance Plan**                     | Comprehensive Coverage Plan        |
| **Policy Expiry Date**                 | March 25, 2027                     |

---

### 🎯 **Now you're all set to run the AI-powered insurance extraction app!** 🚀🔥  

