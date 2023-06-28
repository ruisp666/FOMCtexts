import pickle
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.vectorstores import FAISS
import logging

# Create a global logger
logger = logging.getLogger('logger')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


class DocsToVectorStore:
    def __init__(self,
                 folder,
                 document_splitter=SentenceTransformersTokenTextSplitter(
                     model_name='sentence-transformers/all-mpnet-base-v2', chunk_overlap=5),
                 model_kwargs=None,
                 encode_kwargs=None,
                 embeddings=None,
                 loader=None):

        """
        Initialize the DocsToVectorStore class.

        Parameters
        ----------
        folder : str
            The folder path containing the documents to be processed.
        document_splitter : object, optional
            An instance of a document splitter class. Default is None.
        model_kwargs : dict, optional
            Additional keyword arguments for the sentence transformer model. Default is None.
        encode_kwargs : dict, optional
            Additional keyword arguments for encoding document embeddings. Default is None.
        embeddings : langchain.embeddings
            An instance of an embeddings class. Default is
        """
        if encode_kwargs is None:
            encode_kwargs = {'normalize_embeddings': False}
        if model_kwargs is None:
            model_kwargs = {'device': 'mps'}
        if embeddings is None:
            embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2',
                                               model_kwargs=model_kwargs,
                                               encode_kwargs=encode_kwargs)
        if  loader is None:
            loader = PyPDFDirectoryLoader(folder)
        self.folder = folder
        self.model_kwargs = model_kwargs
        self.encode_kwargs = encode_kwargs
        self.loader = loader
        self.document_splitter = document_splitter
        self.embeddings = embeddings
        self.docs = None
        self.processed_docs = None
        self.vector_store = None

        logger.info('DocsToVectorStore initialized with the following parameters:')
        logger.info(f'folder: {folder}')
        logger.info(f'document_splitter: {document_splitter}')
        logger.info(f'model_kwargs: {model_kwargs}')
        logger.info(f'encode_kwargs: {encode_kwargs}')
        logger.info(f'embeddings: {embeddings}')

    def load_docs(self):
        """
        Load the documents from the folder.

        Returns
        -------
        self : object
            The DocsToVectorStore instance.
        """
        logger.info('Loading documents...')
        self.docs = self.loader.load()
        logger.info(f'{len(self.docs)} Documents loaded.')
        return

    def split_docs(self):
        """
        Split the loaded documents into smaller chunks.

        Returns
        -------
        self : object
            The DocsToVectorStore instance.
        """
        logger.info('Splitting documents...')
        self.processed_docs = self.document_splitter.split_documents(self.docs)
        logger.info(f'Split into {len(self.processed_docs)} documents')

    def embed_docs(self):
        """
        Encode the document chunks into embeddings and create a vector store.

        Returns
        -------
        self : object
            The DocsToVectorStore instance.
        """
        logger.info('Embedding documents...')
        self.vector_store = FAISS.from_documents(self.processed_docs, self.embeddings)
        logger.info('Documents embedded.')

    def save_docstore(self, folder_path):
        """
        Save the vector store to the specified folder path.

        Parameters
        ----------
        folder_path : str
            The folder path to save the vector store.
        """
        logger.info(f'Saving vector store in {folder_path}...')
        self.vector_store.save_local(folder_path=folder_path, index_name='index')
        logger.info('Vector store saved.')

    def from_docs_to_store(self, vector_store_path):
        """
        Build the vector store from the loaded documents.

        Parameters
        ----------
        vector_store_path : str
            The folder path to save the vector store.
        """
        self.load_docs()
        self.split_docs()
        self.embed_docs()
        self.save_docstore(folder_path=vector_store_path)


class VectorMall(DocsToVectorStore):
    """
    Aggregates Docstores according to groups of stores from groups of documents
    """

    def __init__(self, docs_folder, grouped_docs):
        super().__init__(docs_folder)
        self.docs_mall = []
        self.grouped_docs = grouped_docs

    def split_into_stores(self):
        for docstore_timestamp in self.grouped_docs.keys():
                self.docs_mall.append(DocsToVectorStore(folder=self.folder,
                                                        loader=PyPDFDirectoryLoader(self.folder, glob=f'*{docstore_timestamp}*.pdf'),
                                                         model_kwargs={'device':'mps'}))




