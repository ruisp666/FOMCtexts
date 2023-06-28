import logging

log = logging.getLogger('filter methods')
logging.basicConfig(level=logging.INFO)


def filter_docs_by_meta(docs, filter_dict):
    """
    Filter documents by multiple parameters
    Parameters:
        docs : List[langchain.schema.Document]
        filter_dict :  Dict[str, Any]

    Returns: List of filtered documents

    Examples:
        docs = [langchain.schema.Document(metadata={'a': 1, 'b': 2}, text='text1')
                langchain.schema.Document(metadata={'a': 1, 'b': 3}, text='text2')]
        filter_dict = {'a': 1}
        filter_docs_by_meta(docs, filter_dict)
        [langchain.schema.Document(metadata={'a': 1, 'b': 2}, text='text1')]

        docs = [langchain.schema.Document(metadata={'a': 1, 'b': 2}, text='text1')
                langchain.schema.Document(metadata={'a': 1, 'b': 3}, text='text2')]
        filter_dict = {'a': 1, 'b': 2}
        filter_docs_by_meta(docs, filter_dict)
        [langchain.schema.Document(metadata={'a': 1, 'b': 2}, text='text1')]

    """
    filtered_docs = []
    for doc in docs:
        append = True
        for key, value in filter_dict.items():
            if doc.metadata[key] != value:
                append = False
                break
        if append:
            filtered_docs.append(doc)
    return filtered_docs


def search_with_filter(vector_store, query, filter_dict, target_k=5, init_k=100, step=50):
    """
    Expand search with filter until reaching at least a pre-determined number of documents.
    ----------
    Parameters
        vector_store : langchain.vectorstores.FAISS
            The FAISS vector store.
        query : str
            The query to search for.
        filter_dict :  Dict[str, Any]
            The parameters to filer for
        target_k : int
            The minimum number of documents desired after filtering
        init_k : int
            The top-k documents to extract for the initial search.
        step : int
            The size of the step when enlarging the search.

    Returns: List of at least target_k Documents for post-processing

    """
    context = filter_docs_by_meta(vector_store.similarity_search(query, k=init_k), filter_dict)
    while len(context) < target_k:
        log.info(f'Context contains {len(context)} documents')
        log.info(f'Expanding search with k={init_k}')
        init_k += step
        context = filter_docs_by_meta(vector_store.similarity_search(query, k=init_k), filter_dict)
    log.info(f'Done. Context contains {len(context)} Documents matching the filtering criteria')
    return context

