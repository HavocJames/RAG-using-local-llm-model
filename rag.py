from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import Chroma
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
#from langchain.memory import ConversationBufferMemory
from huggingface_hub import hf_hub_download

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

store = LocalFileStore("./CacheBackedEmbeddings/")
underlying_embeddings = HuggingFaceInstructEmbeddings(
    model_name="hkunlp/instructor-large",
    model_kwargs={"device": "cpu"},)

cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings, store, namespace="hkunlp/instructor-large")

loader = PyPDFLoader("/home/ai-project/Report_body_final.pdf")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

vectorstore = Chroma.from_documents(documents=all_splits, embedding=cached_embedder)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

MODELS_PATH = "./models"

model_path = hf_hub_download(   
    repo_id= "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
    filename="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    resume_download=True,
    cache_dir=MODELS_PATH,)

kwargs = {
    "model_path": model_path,
    "temperature": 0.7,
    "top_p" : 1,
    "n_ctx": 2048,
    "callback_manager" : callback_manager,
    "max_tokens": 2048,
    "verbose" : True, 
    "n_batch": 512,  # set this based on your GPU & CPU RAM
}
prompt_template = "<s>[INST] "+"""You are a helpful robot assistant, you will answer user questions by thinking step by step.
Give out short answers. do not ask questions. find the answer from the document provided. 
Human: {question}
Assistant:"""+" [/INST]"

prompt = PromptTemplate(input_variables=["question"], template=prompt_template)

# Make sure the model path is correct for your system!
llm = LlamaCpp(**kwargs)

#memory = ConversationBufferMemory(memory_key=memory_key, llm=llm)

# Chain
llm_chain = LLMChain(llm=llm, prompt=prompt)

while True:
    question = input("\nEnter a query: ")
    docs = vectorstore.similarity_search(question)
    result = llm_chain(docs)
    print(result)