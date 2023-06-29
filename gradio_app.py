import logging

from langchain import PromptTemplate, LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
import gradio as gr
import json

from FOMCTexts.prompts import PROMPT_EXTRACT_DATE, PROMPT_FED_ANALYST
from FOMCTexts.filterminutes import search_with_filter

# ToDo: Default to the latest date.
# ToDo: Allow for comparison between two dates.
# ToDo: Allow for aggregations.

# --------------------------Load environment variables--------------------------#
_ = load_dotenv(find_dotenv('.env.rtf'))

# --------------------------Load the sentence transformer and the vector store--------------------------#
model_name = 'sentence-transformers/all-mpnet-base-v2'
model_kwargs = {'device': 'mps'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
vs = FAISS.load_local("MINUTES_FOMC_HISTORY", embeddings)

# --------------------------Import the prompt and define the chain to extract the date in json format---#
PROMPT_DATE = PromptTemplate.from_template(PROMPT_EXTRACT_DATE)
date_extractor = LLMChain(prompt=PROMPT_DATE, llm=ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo'))

# --------------------------Import the prompt and define the qa chain for answering queries--------------------------#
PROMPT_ANALYST = PromptTemplate.from_template(PROMPT_FED_ANALYST)
fed_chain = load_qa_chain(llm=ChatOpenAI(model_name='gpt-3.5-turbo'), chain_type='stuff', prompt=PROMPT_ANALYST)


def get_chain(query):
    """
    Detects the date, computes similarity, and answers the query using
    only documents corresponding to the desired date embedded in the date
    Parameters
    ----------
    query : str
        Query to be answered.

    Returns
    -------

    """

    logging.info('Extracting the date in numeric format..')
    date_response = date_extractor.run(query)
    print(date_response)
    if date_response != 'False':
        filter_date = json.loads(date_response)

        logging.info(f'Date parameters retrieved: {filter_date}')
        logging.info('Running the qa with filtered context..')
        filtered_context = search_with_filter(vs, query, init_k=200, step=300, target_k=7, filter_dict=filter_date)

        logging.info(20 * '-' + 'Metadata for the documents to be used' + 20 * '-')
        for doc in filtered_context:
            logging.info(doc.metadata)
    else:
        logging.info('No date elements found. Running the qa without filtering can output incorrect results.')
        filtered_context = vs.similarity_search(query, k=7)
    return fed_chain({'input_documents': filtered_context, 'question': query})['output_text']


if __name__ == '__main__':
    demo = gr.Interface(fn=get_chain,
                        inputs=gr.Textbox(lines=2, placeholder="Enter your query", label='Your query'),
                        description='Query the public database in FRED from 1913-2023',
                        outputs='text',
                        title='Chat with the FOMC meeting minutes',
                        examples=['What was the economic outlook from the staff presented in the meeting '
                                  'of April 2009 with respect to labour market developments and industrial production?'
                                  'Write three paragraphs about this topic. ',
                                  'Who were the voting members present in the meeting on April 2009?'],
                        )
    demo.launch()
