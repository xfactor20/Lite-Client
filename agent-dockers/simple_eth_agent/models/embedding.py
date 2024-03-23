import os

from llama_index import ServiceContext, VectorStoreIndex, load_index_from_storage, StorageContext
from llama_index.embeddings import OllamaEmbedding as LlamaIndexOllamaEmbeddings

from langchain.embeddings import OllamaEmbeddings as LangChainOllamaEmbeddings

cache_dir_parent_path = "./cache_dirs"

contracts_metadata_cache_dir = os.path.join(cache_dir_parent_path,
                                            'contracts_metadata_cache')


def langchain_embeddings_factory():
    return LangChainOllamaEmbeddings()  # uses llama2


def build_llamaindex_index(documents):
    service_context = ServiceContext.from_defaults(embed_model=LlamaIndexOllamaEmbeddings(model_name="llama2"),
                                                   llm=None, chunk_size=4096)

    if os.path.isdir(contracts_metadata_cache_dir):
        # rebuild storage context
        storage_context = StorageContext.from_defaults(persist_dir=contracts_metadata_cache_dir)
        # load index
        index = load_index_from_storage(storage_context, service_context=service_context, index_id="vector_index")
    else:
        index = VectorStoreIndex.from_documents(
            documents,
            service_context=service_context,
        )
        index.set_index_id("vector_index")
        index.storage_context.persist(contracts_metadata_cache_dir)

    return index
