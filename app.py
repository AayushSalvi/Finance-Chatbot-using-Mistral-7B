#from haystack.nodes import EmbeddingRetriever, AnswerParser, PromptModel, PromptModel, PromptTemplate

from haystack_integrations.document_stores.weaviate.document_store import WeaviateDocumentStore
from haystack import Pipeline
from haystack import Pipeline
from haystack.components.writers import DocumentWriter
from haystack_integrations.components.retrievers.weaviate.embedding_retriever import WeaviateEmbeddingRetriever
from haystack.components.embedders import SentenceTransformersDocumentEmbedder,SentenceTransformersTextEmbedder
from haystack_integrations.components.generators.mistral import MistralChatGenerator
from haystack.components.generators.utils import print_streaming_chunk
from haystack.components.generators.chat import HuggingFaceLocalChatGenerator
from haystack.components.builders.chat_prompt_builder import ChatPromptBuilder
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret
from haystack.components.builders.answer_builder import AnswerBuilder

#from model_add import LlamaCPPInvocationLayer

from fastapi import FastAPI, Depends, HTTPException, status, Request, Form, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
import uvicorn
import json
import re
import os 
import sys
import pathlib
folder_path = pathlib.Path().parent.resolve()
sys.path.append(os.path.join(folder_path, '../'))
from dotenv import load_dotenv
load_dotenv()

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

print("Import Successfully")

app = FastAPI()

# Configure templates
templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")

def get_result(query):
    document_store = WeaviateDocumentStore(
    url= 'http://localhost:8080/v1'
    )

    # prompt_template = """"Given the provided Documents, answer the Query. Make your answer detailed and long\n
    #                                             Query: {query}\n
    #                                             Documents: {join(documents)}
    #                                             Answer: 
    #                                         """,
                                            
    # print("Prompt Template: ", prompt_template)
    # def initialize_model():
    #     return PromptModel(
    #         model_name_or_path="model/mistral-7b-instruct-v0.1.Q4_K_S.gguf",
    #         invocation_layer_class=LlamaCPPInvocationLayer,
    #         use_gpu=False,
    #         max_length=512
    #     )
    # model = initialize_model()
    llm = MistralChatGenerator(streaming_callback=print_streaming_chunk, model='mistral-small',api_key=Secret.from_env_var("MISTRAL_API_KEY"))
    print("LLM: ", llm)

    embedder = SentenceTransformersTextEmbedder(
                        model="sentence-transformers/all-MiniLM-L6-v2"
                        )
    print("embedder:", embedder)

    retriever = WeaviateEmbeddingRetriever(document_store=document_store)

    print("Retriever: ", retriever)

    builder = ChatPromptBuilder(variables=['documents'])

    print("Builder :", builder)

    ans_builder = AnswerBuilder()

    query_pipeline = Pipeline()
    query_pipeline.add_component(instance=embedder,name="embedder")
    query_pipeline.add_component(instance=retriever, name="Retriever")
    query_pipeline.add_component(instance= builder, name="prompt_builder")
    query_pipeline.add_component(instance=llm, name="llm")

    query_pipeline.connect("embedder","Retriever.query_embedding")
    query_pipeline.connect("Retriever","prompt_builder.documents")
    query_pipeline.connect("prompt_builder","llm")
    
    print({"promt_builder.documents"})
    messages = [ChatMessage.from_user("Here are some List of Dcuments {{documents[0].content}}, Answer them in long and detailed format")]

    print("Query Pipeline: ", query_pipeline)

    #query = query , params={"Retriever" : {"top_k": 5}}
    #{"embedder": {"template_variables":{"query": query}, "template": messages}}
    json_response = query_pipeline.run({
                                        "embedder": {"text": query},
                                        "prompt_builder": {"template_variables": {"query": query}, "template": messages},
                                        "llm": {"generation_kwargs": {"max_tokens": 500}}
                                        }
                                       )

    # json_response = query_pipeline.run(
    # {
    #     "embedder": {"query": query},
    #     "prompt_builder": {"query": query},
    #     "answer_builder": {"query": query},
    # }
    # )
    import json
    #print("Answer: ", json_response['llm']['replies'][0].content)
    answers = json_response['llm']['replies'][0].content
    for ans in answers:
        answer = ans
        break

    # Extract relevant documents and their content
    # documents = json_response['documents']
    # document_info = []

    # for document in documents:
    #     content = document.content
    #     document_info.append(content)

    # Print the extracted information
    print("Answer:")
    print(answer)
    # Split the text into sentences using regular expressions
    sentences = re.split(r'(?<=[.!?])\s', answers)

    # Filter out incomplete sentences
    complete_sentences = [sentence for sentence in sentences if re.search(r'[.!?]$', sentence)]

    # Rejoin the complete sentences into a single string
    updated_answer = ' '.join(complete_sentences)

    # relevant_documents = ""
    # for i, doc_content in enumerate(document_info):
    #     relevant_documents+= f"Document {i + 1} Content:"
    #     relevant_documents+=doc_content
    #     relevant_documents+="\n"

    # print("Relevant Documents:", relevant_documents)

    return updated_answer

@app.get("/")   
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/get_answer")
async def get_answer(request: Request, question: str = Form(...)):
    print(question)
    answer = get_result(question)
    response_data = jsonable_encoder(json.dumps({"answer": answer}))
    res = Response(response_data)
    return res

# if __name__ == "__main__":
#     uvicorn.run("app:app", host='0.0.0.0', port=8001, reload=True)