# Massey University Policy RAG
RAG approach LLM using Mistral-7B-instruct-v0.2 with Llama-index LangChain framework.

The methodology procedure adopted is as follows:

1. Splitting of the documents by chunk size, Hugging Face embeddings, and storing the obtained vector into a vector database. These were achieved using llama-index as the framework and the code procedure is presented in func.py file inside utils folder.
2. Using Mistral 7B instruct as the large language model (LLM). This is implemented by downloading the quantized form of the model from huggingface repo along with the applicable tokenizer and embedding framework.
3. Implementation of retrieval augmentation (RAG) approach by creating a link between the document, the LLM model, and the desire system behaviour and query response using llama-index framework (func.py)
4. Development of frontend application for the chatbot using streamlit library (main.py)
5. The required dependencies for the replication of the work were documented in the requirements.txt file

Work-in-Progress: Use Openai API to generate synthetic summaries and questions for each chunk, and document
