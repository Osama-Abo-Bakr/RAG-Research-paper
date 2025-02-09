# **Research Paper QA Assistant (RAG) 📚**

## **Overview**

This project is a **Research Paper QA Assistant** built using **Retrieval-Augmented Generation (RAG)**. It allows users to upload **PDF research papers**, generates a **text summary**, and enables an **interactive Q&A** system to answer questions about the paper's content.

## **Features**

✅ Accepts multiple **PDF** research papers (up to 5)  
✅ Converts PDFs into **images** for better processing  
✅ Extracts and summarizes **key information** in Markdown format  
✅ Enables **conversational Q&A** with a research assistant  
✅ Uses **FAISS vector search** for efficient retrieval  
✅ Supports **OpenAI models (GPT-4o)** for text summarization and Q&A  
✅ Displays uploaded PDFs and extracted text for quick reference  
✅ Uses **bounding boxes to highlight relevant text chunks**  
✅ Supports **multi-modal analysis** (text + images)  
✅ Downloads a summary as a **PDF or Markdown**  

## **Tech Stack**

- **Python**: Core programming language  
- **Streamlit**: Web-based UI  
- **OpenAI API**: LLM-based summarization, Q&A, and chunk extraction  
- **FAISS**: Vector database for retrieval  
- **LangChain**: Handles document parsing, embeddings, and LLM integration  
- **Docling**: Converts PDFs into structured text  
- **PyMuPDF (fitz)**: Extracts images from PDFs  
- **OpenCV (cv2)**: Image processing and bounding box visualization  
- **pdfkit & markdown2**: Converts Markdown summaries into PDF  

## **Installation**

### **1. Clone the Repository**

```bash
git clone https://github.com/Osama-Abo-Bakr/RAG-Research-paper.git 
cd RAG-Research-paper
```

### **2. Create a Virtual Environment**

```bash
python -m venv venv  
source venv/bin/activate  # For macOS/Linux  
venv\Scripts\activate     # For Windows  
```

### **3. Install Dependencies**

```bash
pip install -r requirements.txt  
```

### **4. Set Up Environment Variables**

Copy `.env.example` to `.env` and update your API keys:

```bash
cp .env.example .env  
```

Modify `.env` with your credentials:

```
OPENAI_API_KEY=your_openai_api_key 
LANDING_API_KEY=your_landing_api_key 
```

## **Usage**

### **Run the Application**

```bash
streamlit run app.py 
```

The app will launch in your web browser.

### **Upload Research Papers**

1. Upload one or more **PDF research papers** from the sidebar.  
2. The system converts each **PDF page into an image** for processing.  
3. A **summary** of the paper is automatically generated.  
4. Users can ask **questions** in natural language.  
5. The **uploaded PDFs and extracted text** are displayed in the sidebar for reference.  
6. The app highlights the **most relevant text chunks** with **bounding boxes** in the extracted images.  

## **File Structure**

```
📺 your-repo/  
 ├── 📝 app.py           # Main application logic  
 ├── 📝 requirements.txt  # Dependencies  
 ├── 📝 LICENSE           # License file  
 ├── 📝 README.md         # Project documentation  
 ├── 📝 .env.example      # Example environment variables  
 ├── 📝 data/           # PDF and image storage  
 ├── 📝 wkhtmltopdf/      # Helper functions for install as pdf
```

## **License**

This project is licensed under the terms of the **MIT License**. See [LICENSE](LICENSE) for details.

## **Author**

👨‍💻 **Osama Abo-Bakr**  
🔗 [GitHub](https://github.com/Osama-Abo-Bakr)