
# Import required libraries
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig
from dotenv import load_dotenv
import streamlit as st
# Import the required langchain llama index libraries 
from llama_index.prompts.prompts import SimpleInputPrompt
from llama_index.llms import HuggingFaceLLM
from llama_index.embeddings import LangchainEmbedding
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import set_global_service_context
from llama_index import ServiceContext
from llama_index import VectorStoreIndex, SimpleDirectoryReader
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

# Avoid reloading the model each run
@st.cache_resource 
def get_tokenizer_model(name, auth_token):
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(name, cache_dir='./model/', use_auth_token=auth_token)

    model = AutoModelForCausalLM.from_pretrained(name, cache_dir='./model/'
                            , use_auth_token=auth_token, torch_dtype=torch.float16, 
                            quantization_config=bnb_config, device_map='auto') 
    return tokenizer, model

# Load the tokenizer and the LLM model
tokenizer, model = get_tokenizer_model(name, auth_token)

def llm_query():
    # Create a system prompt 
    system_prompt = """<s>[INST] <<SYS>>
    You are an helpful, respectful and honest assistant. Always answer 
    accurately, do NOT fake information. Your answers must not include any harmful, 
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

    # Create an HF LLM using the llama index wrapper 
    llm = HuggingFaceLLM(context_window=4096,
                        max_new_tokens=256,
                        system_prompt=system_prompt,
                        query_wrapper_prompt=query_wrapper_prompt,
                        model=model,
                        tokenizer=tokenizer)

    # Create and dl embeddings instance  
    embeddings=LangchainEmbedding(
        HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    )

    # Create new service context instance to allow llmindex work with huggingface

    service_context = ServiceContext.from_defaults(
        chunk_size=1024,
        chunk_overlap=20,
        llm=llm,
        embed_model=embeddings
    )
  
    #set the service context
    #set_global_service_context(service_context)
    documents = SimpleDirectoryReader('./BCN_4787C.pdf').load_data()
 
    # Create an index
    index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    # Setup index query engine using LLM 
    query_engine = index.as_query_engine()
    return query_engine
