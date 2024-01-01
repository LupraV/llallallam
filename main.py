import os
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from utils.func import llm_query, get_tokenizer_model
import torch
from dotenv import load_dotenv

# Hugging face
load_dotenv()
auth_token = os.environ["HUGGING_FACE_API_KEY"]

query_engine = llm_query()

"""Streamlit implementation"""
# Main title 
st.title('Massey University Policies Helper')

# Text input box for the user
prompt = st.text_input('Type your question here')

# If user hits enter
if prompt:
    response = query_engine.query(prompt)
    st.write(response)

    # Display raw response
    with st.expander('Response'):
        st.write(response)
    # Display source text
    with st.expander('Source Text'):
        st.write(response.get_formatted_sources())


if __name__ == '__main__':
    main()

