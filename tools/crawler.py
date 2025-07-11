import os
from langchain_core.tools import tool
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from firecrawl import FirecrawlApp, ScrapeOptions
from config import FIRECRAWL_API_KEY, CHROMA_DIR, TARGET_WEBSITE

# Where to store DB
persist_directory = os.path.join(CHROMA_DIR, "vitalbridge_db")

# Only build retriever once
retriever_chain = None

def fetch_vitalbridge_docs() -> list[Document]:
    print("Fetching data from Firecrawl SDK...")
    firecrawl_client = FirecrawlApp(api_key=FIRECRAWL_API_KEY)

    result = firecrawl_client.crawl_url(
        url=TARGET_WEBSITE,
        limit=100,
        scrape_options=ScrapeOptions(formats=['markdown', 'html']),
    )

    #print(result)

    docs = []
    for page in result.data:
        content = page.markdown or page.html or ""
        source = (
            page.metadata.get("sourceURL")
            if isinstance(page.metadata, dict) and "sourceURL" in page.metadata
            else TARGET_WEBSITE
        )
        if content:
            docs.append(Document(page_content=content, metadata={"source": source}))

    return docs


def get_retriever_chain():
    global retriever_chain
    if retriever_chain:
        return retriever_chain

    if not os.path.exists(persist_directory):
        print("Indexing website into vector DB...")
        raw_docs = fetch_vitalbridge_docs()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = splitter.split_documents(raw_docs)

        db = Chroma.from_documents(split_docs, embedding=OpenAIEmbeddings(), 
                                   persist_directory=persist_directory)
        #db.persist()
    else:
        db = Chroma(persist_directory=persist_directory, embedding_function=OpenAIEmbeddings())

    retriever = db.as_retriever()
    retriever_chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(), retriever=retriever)
    return retriever_chain


@tool
def vitalbridge_info(query: str) -> str:
    """Use this tool to answer questions about Vitalbridge (绿洲) 
    or the content on https://www.vitalbridge.com."""
    qa = get_retriever_chain()
    return qa.invoke(query)