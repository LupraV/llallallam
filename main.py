import os
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from utils.func import llm_query, get_tokenizer_model
import torch
from dotenv import load_dotenv

#======= hugging face settings ===============
load_dotenv()
auth_token = os.environ["HUGGING_FACE_API_KEY"]

query_engine = llm_query()

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


if __name__ == '__main__':
    main()

