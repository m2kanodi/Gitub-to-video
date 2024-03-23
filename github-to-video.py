from langchain_community.document_loaders import GithubFileLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts.prompt import PromptTemplate
from langchain_community.llms import Ollama


ACCESS_TOKEN = "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"


def explain(documents):
    text_splitter = CharacterTextSplitter(chunk_size=10, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    embedding_function = OllamaEmbeddings()
    SYSTEM_TEMPLATE = """
    You are an experienced senior developer at a Big Tech company. Use this context and try to answer the users questions.
    If the context doesn't contain any relevant information to the question, don't make something up and just say "I don't know":

    <context>
    {context}
    </context>
    Question: {input}
    """
    

    # load it into Chroma
    db = Chroma.from_documents(docs, embedding_function)
    query = """summarize by individual files give  heading say file name give whats it doing in that file include all the files in all the folders.
    Answer in the following format:
    Filename: Function performed by the file.
    Do this for all the files."""
    docs = db.similarity_search(query)
    retriever = db.as_retriever()
    docs_ret = retriever.invoke(query)
    llm = Ollama(model="llama2")
    PROMPT = PromptTemplate(input_variables=["context","input"], template=SYSTEM_TEMPLATE)
    doc_chain = create_stuff_documents_chain(llm, PROMPT)
    return doc_chain.invoke({
    "context" : docs_ret,
    "input": query
    })




    # print results
    
    # print(docs[0].page_content)
    # return docs[0].page_content
    



url = 'karthikkrishna1/HTV8'
if (url):
    loader = GithubFileLoader(
        repo=f'{url}',  # the repo name
        access_token = ACCESS_TOKEN,
        github_api_url="https://api.github.com",
        file_filter=lambda file_path: file_path.endswith('.js') and 'node_modules' not in file_path
    )
    documents = loader.load()
    print(explain(documents))