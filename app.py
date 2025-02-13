# ------------ Accept Multi PDFs Files ----------------------------------- #
# ------------------------------------------------------------------------ #
import os
import tempfile
import base64
import warnings
import json
from fpdf import FPDF
import fitz
import cv2
import requests
import streamlit as st
import numpy as np
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
from openai import OpenAI
from collections import defaultdict
from langchain.schema import Document
from docling.document_converter import DocumentConverter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from streamlit_pdf_viewer import pdf_viewer

# Set environment variables
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load environment variables once at startup
load_dotenv()


def pdf_to_images(upload_pdfs):
    """
        Convert a PDF file to a list of images, where each page of the PDF is represented as an image.

        This function saves the uploaded PDF file to a temporary location, converts each page into an
        image using PyMuPDF (fitz), and stores the images as NumPy arrays. After processing, the
        temporary file is deleted.

        Args:
            upload_pdfs (File): An uploaded file-like object representing the PDF to be converted.

        Returns:
            list: A list of NumPy arrays, where each array corresponds to an image of a PDF page.

        Raises:
            Exception: Logs errors encountered during PDF processing or image conversion.
            PermissionError: Logs errors if the temporary file cannot be deleted.

        Notes:
            - The function temporarily saves the uploaded PDF file to the current working directory.
            - Ensure that the uploaded PDF has a valid format and is accessible.
            - Adjust the DPI value in the `get_pixmap` call to control the resolution of the output images.
    """
    pdfs_images = {}
    for upload_pdf in upload_pdfs:
        images = []
        pdf_document = None
        try:
            # Open the PDF using PyMuPDF
            pdf_document = fitz.open(upload_pdf)
            for page_number in range(len(pdf_document)):
                page = pdf_document[page_number]
                # pix = page.get_pixmap(dpi=50)  # Adjust DPI as needed
                pix = page.get_pixmap(dpi=200)  # Best Resolution For Me
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append(np.array(img))  # Store as NumPy array

            pdfs_images[upload_pdf] = images
        except Exception as e:
            print(f"Error processing PDF {upload_pdf}: {str(e)}")
    
    print("🚀 PDFs converted to images successfully!")
    return pdfs_images



def summarize_text(text):
    """
    Summarize a given text using the OpenAI chat API.
    Args:
        text (str): The text to summarize.

    Returns:
        str: The summary of the text in Markdown format, or an error message if the API call fails.

    Raises:
        Exception: If the API call fails.
    """
    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes each research paper in Markdown format."},
                {"role": "user", "content": f"Summarize the following text in Markdown format:\n{text}"}
            ],
            temperature=0.5
        )
        summary = response.choices[0].message.content
        return summary

    except Exception as e:
        return f"An error summarization: {e}"


def load_vector_db(pdf_paths):
    """
    Loads a vector database from a list of PDF files.

    Args:
        pdf_paths (list[str]): The paths to the PDF files to load.

    Returns:
        tuple: A tuple containing the loaded VectorStore and the text from the PDFs.

    Raises:
        Exception: If the vector database can't be loaded.
    """
    try:
        docling_converter = DocumentConverter()
        embeddings = OpenAIEmbeddings()
        
        final_text = ""
        page_docs = defaultdict(lambda: {"texts": [], "tables": []})


        # Load and process each PDF file
        for pdf_path in pdf_paths:
            file_id = pdf_path
            
            # Docling Conversion            
            loader = docling_converter.convert(pdf_path)
            documents = loader.document
            documents_dict = documents.export_to_dict()
            documents_markdown = documents.export_to_markdown()
            
            # Save full markdown
            final_text += documents_markdown + "\n\n"
            
            
            # 1️⃣ Extract Text Data and group by page number
            print('1️⃣ 📖 Extracting Text Data...')
            for text_entry in documents_dict["texts"]:
                text = text_entry["text"].strip()
                page_no = text_entry["prov"][0]["page_no"]
                                                
                key = f"{file_id}_page_{page_no}"
                page_docs[key]["texts"].append(text)
            
            # 2️⃣ Extract Table Data and group by page number
            print('2️⃣ 📖 Extracting Table Data...')
            for table in documents_dict["tables"]:
                page_no = table["prov"][0]["page_no"]
                
                # Convert table cells to a pipe-separated string
                table_text = "TABLE:\n"
                for cell in table["data"]["table_cells"]:
                    table_text += f"{cell['text']}  |  "
                    
                key = f"{file_id}_page_{page_no}"
                page_docs[key]["tables"].append(table_text)
        
        # Now merge text and table content for each key
        final_documents = []
        for key in sorted(page_docs.keys()):
            merged_content = ""
            if page_docs[key]["texts"]:
                merged_text = " ".join(page_docs[key]["texts"])
                merged_content += "TEXT:\n" + merged_text + "\n"
            if page_docs[key]["tables"]:
                merged_tables = "\n".join(page_docs[key]["tables"])
                merged_content += "TABLES:\n" + merged_tables + "\n"
            
            # Extract file_id and page_no from the key
            file_id, _, page_no = key.partition("_page_")
            page_no = int(page_no)
            
            
            final_documents.append(
                Document(
                    page_content=merged_content,
                    metadata={"file_id": file_id, "page_no": page_no}
                )
            )
            
        # # Build the vector store from the combined documents.
        vectorstore = FAISS.from_documents(final_documents, embeddings)

        print("🔥 Finish Creating VectorDB.")
        return vectorstore, final_text
    
    except Exception as e:
        print(f"Error loading vector DB: {e}")
        return None, None
    

def load_qa_chain(vectorstore):
    """
    Loads a conversational retrieval chain from a vectorstore.

    Given a vectorstore, loads a conversational retrieval chain using the following steps:

    1. Loads a retriever from the vectorstore using FAISS.
    2. Loads an LLM from the OpenAI API.
    3. Creates a PromptTemplate using a template string.
    4. Creates a ConversationalRetrievalChain from the retriever and LLM.

    Args:
        vectorstore (FAISS): The vector store containing the document embeddings.

    Returns:
        ConversationalRetrievalChain: The conversational retrieval chain.
    """
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5})
    llm = ChatOpenAI(model="gpt-4o")

    template = """
    You are an AI research assistant specializing in research_field.  
    Your task is to answer questions about the research papers: "{papers_titles}".  

    Use the following context from the paper to provide an accurate response in markdown format (highly structured):  
    {context}  

    Question: {question}  

    Answer the question strictly based on the provided context. If the context is insufficient, state that more information is needed.
    """

    prompt = PromptTemplate(
        input_variables=["papers_titles", "context", "question"], 
        template=template
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt, "document_variable_name": "context"},
    )


def display_pdf_in_sidebar(pdf_uploader):
    """
    Displays a PDF in the Streamlit sidebar using an iframe.

    Args:
        pdf_uploader (UploadedFile): The PDF file to display.

    Returns:
        None
    """
    if pdf_uploader is not None:
        try:
            # pdf_data = pdf_uploader.getvalue()
            # base64_pdf = base64.b64encode(pdf_data).decode("utf-8")

            # # Display PDF in the sidebar
            # pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600px" type="application/pdf"></iframe>'
            # st.markdown(pdf_display, unsafe_allow_html=True)
            
            pdf_viewer(pdf_uploader.getvalue())

        except Exception as e:
            st.sidebar.error(f"An error occurred while displaying the PDF: {e}")

def markdown_to_pdf(content, file_name="paper_summary.pdf"):
    """
    Converts Markdown content to a PDF file.

    This function takes Markdown content as input, processes it line by line,
    and generates a PDF file with formatted titles, subtitles, and text. The
    PDF is saved with the name "paper_summary.pdf".

    Args:
        content (str): The Markdown content to be converted into a PDF.

    Returns:
        str: The name of the output PDF file.

    Notes:
        - Title lines starting with "# ", "## ", and "### " are formatted as
          titles and subtitles of varying sizes.
        - Text lines with "**" are treated as bold text.
        - Lines starting with "- " are treated as bullet points.
        - Unsupported characters are replaced to avoid encoding errors.
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    def safe_text(text):
        """Encode text to avoid errors with unsupported characters."""
        return text.encode('latin-1', 'replace').decode('latin-1')  # Replaces unsupported characters

    def add_title(text, size=16):
        pdf.set_font("Arial", style="B", size=size)
        pdf.cell(0, 10, txt=safe_text(text), ln=True, align="C")
        pdf.ln(5)

    def add_subtitle(text, size=14):
        pdf.set_font("Arial", style="B", size=size)
        pdf.cell(0, 7, txt=safe_text(text), ln=True, align="L")
        pdf.ln(4)

    def add_text(text):
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 6, txt=safe_text(text))
        pdf.ln(2)

    lines = content.split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            pdf.ln(3)
            continue

        if line.startswith("# "):
            add_title(line[2:])
        elif line.startswith("## "):
            add_subtitle(line[3:], size=14)
        elif line.startswith("### "):
            add_subtitle(line[4:], size=12)
        elif "**" in line:
            line = line.replace("**", "")  # Remove Markdown bold markers
            add_subtitle(line, size=12)
        elif line.startswith("- "):
            add_text("• " + line[2:])
        else:
            add_text(line)

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(temp_file.name, "F")
    return temp_file.name


def extract_text_from_pdf(image_np):
    """
    Extracts text, tables, and figures from an image of a PDF page using an external API.

    This function takes a NumPy array representation of an image, converts it to a PIL Image,
    and then to a file-like object. It sends the image to an external document analysis API
    to extract text, tables, and figures.

    Args:
        image_np (np.ndarray): A NumPy array representing the image of a PDF page.

    Returns:
        dict: A JSON response from the API. If successful, it contains extracted information
              such as text, tables, and figures. In case of an error, it returns an error message.
    """
    image_pil = Image.fromarray(image_np)

    img_byte_arr = BytesIO()
    image_pil.save(img_byte_arr, format="PNG")  # Save in PNG format
    img_byte_arr.seek(0)  # Move cursor to start of the file

    # API request
    url = "https://api.landing.ai/v1/tools/document-analysis"
    files = {"image": ("image.png", img_byte_arr, "image/png")}
    data = {
        "parse_text": True,
        "parse_tables": True,
        "parse_figures": True,
        "summary_verbosity": "normal",
        "caption_format": "markdown",
        "response_format": "json",
        "return_chunk_crops": False,
        "return_page_crops": False
    }

    headers = {
        "Authorization": f"Basic {os.getenv('LANDING_API_KEY')}"  # Replace with your actual API key
    }

    response = requests.post(url, files=files, data=data, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"API request failed with status {response.status_code}: {response.text}"}
    
    
def get_best_chunks(user_query, answer, image):
    """
    Uses the OpenAI chat API to get the most relevant text chunks that best support the answer and query.

    Args:
        user_query (str): The user's query.
        answer (str): The answer provided by the user.
        image (np.ndarray): The image of the PDF page.

    Returns:
        dict: A JSON response with the most relevant text chunks as a list of dictionaries, each containing
              a "summary" key with the text and a "bbox" key with the bounding box coordinates in the format
              [x, y, w, h]. If the API call fails, it returns an error message.
    """
    json_data = extract_text_from_pdf(image)
    
    chunks = json_data.get("data", {}).get("pages", [{}])[0].get("chunks", [])
    
    if not chunks:
        return {"error": "No chunks found in the JSON data."}


    prompt = f"""
    You are an intelligent assistant that helps extract the most relevant text chunks from a research paper.
    
    User Query: {user_query}
    Answer Provided: {answer}
    
    Below are text chunks extracted from the document:
    {chunks}
    
    Select the most relevant chunks that best support the answer and query.
    Provide the selected chunks in JSON format as:
    {{
        "best_chunks": [
            {{"summary": "Chunk Text", "bbox": [x, y, w, h]}},
            ...
        ]
    }}
    """
    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": f"{prompt}"}
            ],
            temperature=0.5,
            response_format={ "type": "json_object" }
        )
        summary = response.choices[0].message.content
        return summary

    except Exception as e:
        return f"An error summarization: {e}"

def main():
    st.set_page_config(page_title='RAG QA For Research Papers📃', page_icon="📖", layout="wide")
    st.title("📖 Research Paper QA Assistant (RAG)")
    
    uploaded_files = st.sidebar.file_uploader("📤 Upload a Research Paper (PDF) (Max: 5)", type=['pdf'], accept_multiple_files=True)
    if uploaded_files:
        if len(uploaded_files) > 5:
            st.sidebar.error("⚠️ You can only upload up to 5 PDFs. Please remove extra files.")
            st.stop()
          
        else:
            if "temp_paths" not in st.session_state:    
                st.session_state.temp_paths = []
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                        temp_file.write(uploaded_file.read())
                        st.session_state.temp_paths.append(temp_file.name)

                st.sidebar.success(f"✅ {len(uploaded_files)} PDFs uploaded successfully!")

        # Create a session state dictionary to hold Images
        if "images" not in st.session_state:
            print("\n\n🚀 Converting PDF Into Images...")
            st.session_state.images = pdf_to_images(st.session_state.temp_paths)
            
        # Load vector store and QA chain only if not cached
        if "vector_store" not in st.session_state:
            print("🚀 Initializing Vector Store...")
            st.session_state.names = {uploaded_file.name: uploaded_file for uploaded_file in uploaded_files}
            st.session_state.vector_store, st.session_state.final_text = load_vector_db(st.session_state.temp_paths)

        if "qa_system" not in st.session_state and st.session_state.vector_store:
            print("🚀 Creating QA Chains...")
            st.session_state.qa_system = load_qa_chain(st.session_state.vector_store)

        if "summary" not in st.session_state and st.session_state.final_text:
            print("🚀 Creating Paper Summary...")
            # st.session_state.summary = st.session_state.final_text
            st.session_state.summary = summarize_text(st.session_state.final_text)
            st.session_state.pdf_path = markdown_to_pdf(st.session_state.summary.replace("```markdown", "").replace("```", "").strip())

        if "chat_history" not in st.session_state:
            print("🚀 Creating Chat History...")
            st.session_state.chat_history = []

        # Display summary in the sidebar
        if "summary" in st.session_state:
            with st.expander("Paper Summary:", icon="📑"):
                st.write(st.session_state.summary.replace("```markdown", "").replace("```", "").strip())
            
            try:
                selected_file = st.sidebar.multiselect(label="Select PDFs 📑", 
                                                       options=st.session_state.names.keys(), 
                                                       default=None, 
                                                       max_selections=1)
                
                with st.expander("PDF Preview:", expanded=False, icon="📑"):
                    display_pdf_in_sidebar(st.session_state.names[selected_file[0]])
            except:
                st.warning("Please Select a PDF first. 💾")
        
        if "images" in st.session_state:
            with st.expander("PDF Images Preview:", icon="📸"):
                for i, image in enumerate(st.session_state.images[st.session_state.temp_paths[0]], 1):
                    st.image(image, caption=f"Page {i}")
                    
        
        # User Question Input
        question = st.chat_input("💬 Ask a question about the paper:")
        if question and "qa_system" in st.session_state:
            with st.spinner("Processing your question..."):
                result = st.session_state.qa_system.invoke(
                    {
                        "question": question,
                        "chat_history": st.session_state.chat_history,
                        "papers_titles": [uploaded_file.name for uploaded_file in uploaded_files],
                    }
                )

            st.session_state.chat_history.append((question, result["answer"]))
            # Display chat history
            for chat in st.session_state.chat_history:
                with st.chat_message("user"):
                    st.write(chat[0])

                with st.chat_message("assistant"):
                    st.write(chat[1].replace("```markdown", "").replace("```", "").strip())

            # Display reference documents
            with st.expander("📂 References from the Paper"):
                for i, doc in enumerate(result["source_documents"][:2], 1):
                    # st.write(f"📜 **Document {i}:**")
                    # st.write(doc.page_content)
                    # st.write("---")
                    
                    file_id = doc.metadata["file_id"]
                    page_no = doc.metadata["page_no"]-1
                    
                    # Load image with bounding boxes
                    img = st.session_state.images[file_id][page_no]
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    try:
                        # Extract Best Chunks
                        best_chunks = get_best_chunks(user_query=question,
                                                    answer=result["answer"],
                                                    image=img)
                        
                        try:
                            best_chunks = json.loads(best_chunks)
                        except json.JSONDecodeError:
                            st.error("❌ Failed to parse GPT-4o response. Ensure the model returns valid JSON.")
                            best_chunks = {"best_chunks": []}  
                        
                        if len(best_chunks["best_chunks"]) > 0:
                            st.subheader("🔎 Best Chunks")
                            # st.json(best_chunks)
                            
                            for chunk in best_chunks["best_chunks"]:
                                bbox = chunk["bbox"]
                                x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
                                cv2.rectangle(img, (x, y), (w, h), (0, 255, 0), 1)
                                
                            # Display the image
                            st.image(img, caption=f"Page {doc.metadata['page_no']} with Bounding Boxes", use_column_width=False)
                    except:
                        st.warning("Failed to extract best chunks. Please try again.")    

        # Button to Download Summary
        if "summary" in st.session_state:
        # if st.sidebar.button("📥 Download Summary (PDF)."):
            with open(st.session_state.pdf_path, "rb") as pdf:
                st.sidebar.download_button(
                    label="📥 Download Summary (PDF).",
                    data=pdf,
                    # data=st.session_state.summary,
                    file_name="paper_summary.pdf",
                    mime="application/pdf"
                )
    
    else:
        st.session_state.clear()
        st.warning("Please upload a PDF first. 💾")


if __name__ == "__main__":
    main()