from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# --- Configuration ---
PDF_FILE_PATH = "ISTE\Aurora\Aurora 25' Event Details.pdf"
MODEL_NAME = "distilgpt2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- FastAPI Initialization ---
app = FastAPI()

# --- Load and Initialize Resources ---
print("Loading ...")
loader = PyPDFLoader(PDF_FILE_PATH)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
texts = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(texts, embeddings)


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,# Use auto to automatically select device
    torch_dtype=torch.float16#Load model in 4-bit for reduced memory (quantization)
)


pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=300,
    do_sample=True,
    temperature=0.7,
    top_p=0.85,
    repetition_penalty=1.2,
)
llm = HuggingFacePipeline(pipeline=pipe)

template = """You are a helpful assistant that answers questions based on the provided context. Only use the information from the context to answer. Do not include instructions or unrelated information. If the context does not provide enough information, simply reply, "I don't know."

context:
{context}

Question:
{question}
"""

# --- 4. Create the RetrievalQA Chain ---
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  
    retriever=vectorstore.as_retriever(),
    return_source_documents=False, 
    chain_type_kwargs={
        "prompt": PromptTemplate(
            template=template,
            input_variables=["context", "question"],
        ),
    },
)
# --- API Models ---
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

# --- API Endpoints ---
@app.get("/")
def root():
    return {"message": "Welcome to the PDF Chatbot API!"}

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    question = request.question
    try:
        answer = qa_chain.run(question)
        return QueryResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
