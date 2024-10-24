#rom haystack import EmbeddingRetriever, PreProcessor
# from haystack.document_stores.types. import WeaviateDocumentStore
from haystack.components.converters.pypdf import PyPDFToDocument 
from haystack_integrations.document_stores.weaviate.document_store import WeaviateDocumentStore
from haystack.components.preprocessors import DocumentCleaner
from haystack.components.preprocessors import DocumentCleaner
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack_integrations.components.retrievers.weaviate.embedding_retriever import WeaviateEmbeddingRetriever
from haystack.components.embedders import SentenceTransformersDocumentEmbedder,SentenceTransformersTextEmbedder
from haystack import Pipeline


path_doc = ["data/stockmarket.pdf"] 

document_store = WeaviateDocumentStore(
    url= 'http://localhost:8080/v1'
    )

print("Document Store Done!")

convertor = PyPDFToDocument()
output = convertor.run(path_doc)
docs = output["documents"]

# clean_whitespace=False,
# split_by = "sentence",
# split_length = 300,
# split_respect_sentence_boundary = True

indexing_pipeline  = Pipeline() 
indexing_pipeline.add_component(instance=DocumentCleaner(),name="cleaner")
indexing_pipeline.add_component(instance=DocumentSplitter(split_by="sentence", split_length=1),name="splitter")
indexing_pipeline.add_component(name="embedder", instance=SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2"))
indexing_pipeline.add_component(instance=DocumentWriter(document_store=document_store), name="writer")

indexing_pipeline.connect("cleaner","splitter")
indexing_pipeline.connect("splitter","embedder")
indexing_pipeline.connect("embedder","writer")

indexing_pipeline.run({"documents":docs})
print("indexing pipeline created")


query_pipeline = Pipeline()
query_pipeline.add_component(instance=
                       SentenceTransformersTextEmbedder(
                        model="sentence-transformers/all-MiniLM-L6-v2"
                        ),
                        name="embedder"
                       )
query_pipeline.add_component(instance=WeaviateEmbeddingRetriever(document_store=document_store),name="retriever")

query_pipeline.connect("embedder","retriever")
print("query pipeline created")

query = "What are Stocks?"

result = query_pipeline.run({"embedder": {"text": query}})

print(result["retriever"]["documents"][0])

# preprocessor = DocumentCleaner()
# preprocessed_docs = preprocessor.run(docs)
# doc_splitter = DocumentSplitter(split_by="sentence", split_length=1)
# preprocessed_docs = doc_splitter.run(preprocessed_docs)


#https://docs.haystack.deepset.ai/docs/sentencetransformerstextembedder

# document_store.write_documents(preprocessed_docs)

# retriever = SentenceTransformersDocumentEmbedder(
#     document_store = document_store,
#     model="sentence-transformers/all-MiniLM-L6-v2"
# )













