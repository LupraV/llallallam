# Course-chatbot-using-LLM
A chatbot is developed by RAG approach. The LLM used is llama2 by Meta with llamaIndex framework.

The external libraries used for this work include:

1) streamlit
2) transformer (huggingface)
3) llama-index, and
4) langchain

The procedure adopted in the work as follows:

1) the splitting of the document into chunk size, application of embedding, and storing the obtained vector into a vector database. These were achieved using llama-index as the framework and the code procedure is presented in func.py file inside utils folder.
2) Adopted llama2 with 7 billion parameters as the large language model. This is implemented by downloading the quantized form of the model from huggingface repo along with the applicable tokenizer and embedding framework.
3) implementation of retrieval augmentation (RAG) approach by creating a link between the document, the LLM model, and the desire system behaviour and query response using llama-index framework (func.py)
4) development of frontend application for the chatbot using streamlit library (main.py)
5) The required dependencies for the replication of the work were documented in the requirements.txt file
