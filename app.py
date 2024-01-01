
# Import libraries
import streamlit as st
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig
from dotenv import load_dotenv

# Import the required langchain llama index libraries 
from llama_index.prompts.prompts import SimpleInputPrompt
from llama_index.llms import HuggingFaceLLM
from llama_index.embeddings import LangchainEmbedding
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import set_global_service_context
from llama_index import ServiceContext
from llama_index import VectorStoreIndex, ServiceContext, SimpleDirectoryReader
from pathlib import Path


# Define weights naming variable
name = "model/Mistral-7B-Instruct-v0.2-GGUF"

# Set auth token variable from hugging face 
load_dotenv()

auth_token = os.environ["HUGGING_FACE_API_KEY"]

bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

# avoid reloading model each run
@st.cache_resource
def get_tokenizer_model():
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(name, cache_dir='./model/', use_auth_token=auth_token)

    model = AutoModelForCausalLM.from_pretrained(name, cache_dir='./model/'
                            , use_auth_token=auth_token, torch_dtype=torch.float16, 
                            quantization_config=bnb_config, device_map='auto') 

    return tokenizer, model

# Create a system prompt 
system_prompt = """<s>[INST] <<SYS>>
You are an helpful, respectful and honest assistant. Always answer 
accurately, do NOT . Your answers must not include any harmful, 
unethical, racist, sexist, toxic, dangerous, or illegal content.
Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain 
why instead of answering something incorrect. If you don't know the answer 
to a question, please NEVER share false information.

Your goal is to provide answers relating to Massey University policies information
based on the provided documents, providing also the exact source chunk of texts.<</SYS>>
"""
# query prompt wrapper
query_wrapper_prompt = SimpleInputPrompt("{query_str} [/INST]")

# load model and tokenizer

tokenizer, model = get_tokenizer_model()

# Create a llama-index wrapper for Hugging Face 
llm = HuggingFaceLLM(context_window=4096,
                    max_new_tokens=256,
                    system_prompt=system_prompt,
                    query_wrapper_prompt=query_wrapper_prompt,
                    model=model,
                    tokenizer=tokenizer)

# Create and download embeddings instance  
embeddings=LangchainEmbedding(
    HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
)

# Create new service context instance to allow llama-index work with hugging face
service_context = ServiceContext.from_defaults(
    chunk_size=750,
    overlap=50,
    llm=llm,
    embed_model=embeddings
)

#set the service context
set_global_service_context(service_context)

documents = SimpleDirectoryReader('./MasseyPolicies.zip').load_data()

# Create an index
index = VectorStoreIndex.from_documents(documents)

# Setup index query engine using LLM 
query_engine = index.as_query_engine()

"""Streamlit implementation"""
# Create centered main title 
st.title('Massey university Policies Helper')
# Create a text input box
prompt = st.text_input('Type your question here')

# If the user hits enter
if prompt:
    response = query_engine.query(prompt)
    st.write(response)

    # Display response
    with st.expander('Response'):
        st.write(response)
    # Display source text
    with st.expander('Source Text'):
        st.write(response.get_formatted_sources())

