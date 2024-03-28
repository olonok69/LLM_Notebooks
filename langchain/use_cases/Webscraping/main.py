# Playwright Test was created specifically to accommodate the needs of end-to-end testing.
# Playwright supports all modern rendering engines including Chromium, WebKit, and Firefox. Test on Windows, Linux, and macOS, locally or on CI, headless or headed
# with native mobile emulation of Google Chrome for Android and Mobile Safari.

# https://playwright.dev/docs/intro
# https://playwright.dev/docs/getting-started-vscode
# https://docs.npmjs.com/about-npm

# https://python.langchain.com/docs/integrations/document_loaders/async_html
# uses the aiohttp library to make asynchronous HTTP requests, suitable for simpler and lightweight scraping.

# https://python.langchain.com/docs/integrations/document_loaders/async_chromium
# uses Playwright to launch a Chromium instance, which can handle JavaScript rendering and more complex web interactions.


# https://python.langchain.com/docs/integrations/document_transformers/html2text
# https://github.com/Alir3z4/html2text/

# Beautiful Soup
# https://www.crummy.com/software/BeautifulSoup/
# https://python.langchain.com/docs/integrations/document_transformers/beautiful_soup

# Install pip install html2text playwright beautifulsoup4

# how to GoogleSearchAPIWrapper
# https://programmablesearchengine.google.com/controlpanel/all
# https://console.cloud.google.com/apis/
# https://stackoverflow.com/questions/37083058/programmatically-searching-google-in-python-using-custom-search
# https://cloud.google.com/apis/docs/client-libraries-explained#google_api_client_libraries
# https://github.com/olonok69/LLM_Notebooks/blob/main/langchain/use_cases/Tools_Langchain_OpenAI_Use_cases.ipynb

# Extraction  Langchain
# https://github.com/olonok69/LLM_Notebooks/blob/main/langchain/use_cases/Langchain_OpenAI_Use_cases_Extraction.ipynb


"""
Search: Query to url (e.g., using GoogleSearchAPIWrapper).
Loading: Url to HTML (e.g., using AsyncHtmlLoader, AsyncChromiumLoader, etc).
Transforming: HTML to formatted text (e.g., using HTML2Text or Beautiful Soup).
"""

from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain.chains import create_extraction_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pprint

from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_core.tools import Tool

# Load keys
env_path = ".env"
load_dotenv(env_path)
KEY_OPENAI = os.getenv("KEY_OPENAI")


# Google Search API
"""
os.environ["GOOGLE_CSE_ID"] 
os.environ["GOOGLE_API_KEY"] 
"""

search = GoogleSearchAPIWrapper()

tool = Tool(
    name="google_search",
    description="Search Google for recent results.",
    func=search.run,
)
response = tool.run("who it is prime minister of UK?")

print(response)
input("Press Enter to continue...")

# Load HTML
loader = AsyncChromiumLoader(["https://www.cnn.com"])
# load into Langchain docs
html = loader.load()
print(f"Number of document loaded: {len(html)}")
input("Press Enter to continue...")
print(f"Number of document loaded: {len(html[0].page_content)}")
input("Press Enter to continue...")
print(f"Content document: {html[0].page_content[:500]}")
input("Press Enter to continue...")
print(f"Metadata: {html[0].metadata}")
input("Press Enter to continue...")

# Transform
"""
Scrape text content tags such as <p>, <li>, <div>, and <a> tags from the HTML content:

<p>: The paragraph tag. It defines a paragraph in HTML and is used to group together related sentences and/or phrases.

<li>: The list item tag. It is used within ordered (<ol>) and unordered (<ul>) lists to define individual items within the list.

<div>: The division tag. It is a block-level element used to group other inline or block-level elements.

<a>: The anchor tag. It is used to define hyperlinks.

<span>: an inline container used to mark up a part of a text, or a part of a document.
"""

bs_transformer = BeautifulSoupTransformer()
docs_transformed = bs_transformer.transform_documents(html, tags_to_extract=["span"])

print(docs_transformed[0].page_content)


input("Press Enter to continue...")


## LOADERS

urls = ["https://www.espn.com", "https://lilianweng.github.io/posts/2023-06-23-agent/"]
loader = AsyncHtmlLoader(urls)
docs = loader.load()
for doc in docs:
    print(doc.page_content)
    input("Press Enter to continue...")


## Transformer

urls = ["https://lilianweng.github.io/posts/2023-06-23-agent/"]
loader = AsyncHtmlLoader(urls)
docs = loader.load()
html2text = Html2TextTransformer()
docs_transformed = html2text.transform_documents(docs)
print(docs_transformed[0].page_content)

input("Press Enter to continue...")

## LLM with function calling

KEY_OPENAI = os.getenv("KEY_OPENAI")

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", openai_api_key=KEY_OPENAI)

loader = AsyncChromiumLoader(["https://www.flipkart.com/search?q=iphone"])
docs = loader.load()

bs_transformer = BeautifulSoupTransformer()
docs_transformed = bs_transformer.transform_documents(
    docs, tags_to_extract=["div", "span", "h2", "a", "h1"]
)


splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000, chunk_overlap=0
)
splits = splitter.split_documents(docs_transformed)

schema = {
    "properties": {
        "product_title": {"type": "string"},
        "product_mrp": {"type": "integer"},
        "product_description": {"type": "array", "items": {"type": "string"}},
        "product_reviews_count": {"type": "string"},
    },
    "required": ["product_title", "product_mrp", "product_description"],
}


def extract(content: str, schema: dict):
    return create_extraction_chain(schema=schema, llm=llm).run(content)


for sp in splits:
    extracted_content = extract(schema=schema, content=sp.page_content)
    for x in extracted_content:
        print(x)
        input("Press Enter to continue...")
    break


### Run the web scraper w/ BeautifulSoup

schema = {
    "properties": {
        "news_article_title": {"type": "string"},
        "news_article_summary": {"type": "string"},
    },
    "required": ["news_article_title", "news_article_summary"],
}


def scrape_with_playwright(urls, schema):
    loader = AsyncChromiumLoader(urls)
    docs = loader.load()
    bs_transformer = BeautifulSoupTransformer()
    docs_transformed = bs_transformer.transform_documents(
        docs, tags_to_extract=["div", "span", "h2", "a"]
    )
    print(f"Extracting content with LLM {len(docs_transformed)}")

    # Grab the first 1000 tokens of the site
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=0
    )
    splits = splitter.split_documents(docs_transformed)

    # # Process the first split
    # extracted_content = extract(schema=schema, content=splits[0].page_content)
    # pprint.pprint(extracted_content)
    return splits


urls = ["https://okdiario.com"]
splits = scrape_with_playwright(urls, schema=schema)

for sp in splits:
    print(sp.page_content)
    extracted_content = extract(schema=schema, content=sp.page_content)
    for x in extracted_content:
        pprint.pprint(x)
        input("Press Enter to continue...")
    break
