from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
from langchain.prompts import PromptTemplate
import gradio as gr

# --------------------------Load environment variables--------------------------#
_ = load_dotenv(find_dotenv('.env.rtf'))

# --------------------------Load the vector store--------------------------#
model_name = 'sentence-transformers/all-mpnet-base-v2'
model_kwargs = {'device': 'mps'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
vector_s = FAISS.load_local("fed_minutes_vector_store", embeddings)

# --------------------------Define the prompt--------------------------#
prompt_template = """You are a research analyst at a federal reserve bank and you are trying to answer questions or 
provide colour on statements. Use the following pieces of context to answer the question at the end. Explain the 
rationale behind your answer. If you don't have all the elements to answer the query, say it explicitly.

{context}

Question: {question}
"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
# --------------------------Define the QA model from the chain--------------------------#
ask_me_about_fed_stuff = RetrievalQA.from_chain_type(llm=ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.0),
                                                     chain_type="stuff",
                                                     retriever=vector_s.as_retriever(search_kwargs={"k": 7}),
                                                     return_source_documents=True,
                                                     chain_type_kwargs={'prompt': PROMPT})


def get_chain(query):
    return ask_me_about_fed_stuff({'query': query})['result']


if __name__ == '__main__':
    demo = gr.Interface(fn=get_chain, inputs='text', description='Query the public database in FRED from 1913-2023',
                        examples=[['What is the rationale behind the answer to the question: What is the rationale behind the answer to the question:']], outputs='markdown', title='Query the FOMC minutes')
    demo.launch()
