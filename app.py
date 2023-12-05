
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


# Define variable to hold llama2 weights naming 
name = "meta-llama/Llama-2-7b-chat-hf"

# Set auth token variable from hugging face 
load_dotenv()

auth_token = os.environ["HUGGING_FACE_API_KEY"]

bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

#to avoid reloading of the model at each run
@st.cache_resource
def get_tokenizer_model():
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(name, cache_dir='./model/', use_auth_token=auth_token)

    # # Create model (some changes here)
    # model = AutoModelForCausalLM.from_pretrained(name, cache_dir='./model/'
    #                         , use_auth_token=auth_token, torch_dtype=torch.float16, 
    #                         rope_scaling={"type": "dynamic", "factor": 2}, load_in_8bit=True)

        # Create model (some changes here)

    model = AutoModelForCausalLM.from_pretrained(name, cache_dir='./model/'
                            , use_auth_token=auth_token, torch_dtype=torch.float16, 
                            quantization_config=bnb_config, device_map='auto') 

    return tokenizer, model

# Create a system prompt 
system_prompt = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as 
helpfully as possible, while being safe. Your answers should not include
any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain 
why instead of answering something not correct. If you don't know the answer 
to a question, please don't share false information.

Your goal is to provide answers relating to course information based on the provided document.<</SYS>>
"""
# query prompt wrapper
query_wrapper_prompt = SimpleInputPrompt("{query_str} [/INST]")

# load model and tokenizer

tokenizer, model = get_tokenizer_model()

# Create a HF LLM using the llama index wrapper 
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
    llm=llm,
    embed_model=embeddings
)

#set the service context
set_global_service_context(service_context)

documents = SimpleDirectoryReader('./BCN_4787C.pdf').load_data()

# Create an index
index = VectorStoreIndex.from_documents(documents)

# Setup index query engine using LLM 
query_engine = index.as_query_engine()

"""frontend implementation"""
# Create centered main title 
st.title('🦙 BCN_4787C CourseChatbot')
# Create a text input box for the user
prompt = st.text_input('Input your question here')

# If the user hits enter
if prompt:
    response = query_engine.query(prompt)
    st.write(response)

    # Display raw response object
    with st.expander('Response Object'):
        st.write(response)
    # Display source text
    with st.expander('Source Text'):
        st.write(response.get_formatted_sources())

