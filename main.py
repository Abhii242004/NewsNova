import os
import streamlit as st
import pickle
import time
import logging
import traceback
import requests
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from bs4 import BeautifulSoup # Added BeautifulSoup for HTML parsing

# Import Groq LLM and HuggingFace Embeddings
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
# NOTE: Using langchain.chains directly for compatibility. 
# If this fails due to a linting error, ensure your interpreter is correct.
from langchain_classic.chains import RetrievalQAWithSourcesChain 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents import Document
from groq import RateLimitError # Groq specific rate limit error
from groq import AuthenticationError as GroqAuthError # Groq specific auth error

# Setup logging
LOG_PATH = "app.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler()
    ],
)
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env (especially groq api key)

# Define a custom exception for non-Groq/Auth/RateLimit errors for broader exception handling
class LLMServiceError(Exception):
    pass

# --- NEW FUNCTION FOR HTML PARSING ---
def fetch_and_parse_url(url: str) -> Document | None:
    """Fetches a URL, extracts main text content using BeautifulSoup, and returns a LangChain Document."""
    try:
        logger.info("Fetching URL: %s", url)
        
        # 1. Fetch content with a standard User-Agent header
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        
        # 2. Parse HTML using BeautifulSoup
        soup = BeautifulSoup(resp.content, 'html.parser')
        
        # Extract all readable text, stripping whitespace and using newlines as separators
        text = soup.get_text(separator='\n', strip=True)
        
        # 3. Create LangChain Document
        if text:
            # Use the title if available, otherwise use the URL
            title = soup.title.string if soup.title else url
            return Document(page_content=text, metadata={"source": url, "title": title})
        else:
            logger.warning("No significant text content extracted from URL: %s", url)
            return None

    except requests.exceptions.Timeout:
        logger.error("Timeout occurred while fetching URL: %s", url)
        st.error(f"Timeout occurred while fetching {url}.")
        return None
    except requests.exceptions.RequestException as e:
        logger.error("Request failed for URL %s: %s", url, e)
        st.error(f"Request failed for {url}: {e}")
        return None
    except Exception as e:
        logger.exception("Unexpected error in fetch_and_parse_url for %s", url)
        st.error(f"Unexpected error processing {url}: {str(e)}")
        return None
# --- END NEW FUNCTION ---


# Wrap HuggingFace embeddings (no retry needed as it's local/open-source)
def create_embeddings(docs):
    try:
        # Using a common open-source embedding model that runs locally
        # You may need to install sentence-transformers: pip install sentence-transformers
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return FAISS.from_documents(docs, embeddings)
    except Exception as e:
        st.error(f"⚠️ Unexpected error creating embeddings: {str(e)}")
        logger.exception("Unexpected error in create_embeddings")
        raise

# Initialize LLM with retry logic for Groq RateLimitError
@retry(
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(6),
    retry=retry_if_exception_type(RateLimitError)
)
def get_llm_response(vectorstore, query):
    try:
        # Use ChatGroq instead of OpenAI
        llm = ChatGroq(
            model_name="llama-3.3-70b-versatile",  # Recommended fast and capable Groq model
            temperature=0.9,
        )

        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
        return chain({"question": query}, return_only_outputs=True)
    except GroqAuthError as e:
        st.error("⚠️ Groq API Key is invalid or expired. Please check your API key.")
        logger.error(f"Authentication error: {e}")
        raise
    except RateLimitError as e:
        st.warning("🕒 Groq API rate limit reached. Retrying with backoff...")
        logger.warning(f"Rate limit error: {e}")
        raise
    except Exception as e:
        # Catch any other exception during LLM processing
        st.error(f"⚠️ Unexpected error in LLM processing: {str(e)}")
        logger.exception("Unexpected error in get_llm_response")
        raise

# Reconfigure logging to include file handler (Note: already done above, keeping for robustness)
logger = logging.getLogger(__name__)

st.title("NewsNova 📈")

# API Key Management in Sidebar
st.sidebar.title("⚙️ Configuration")
# Change the environment variable key to GROQ_API_KEY
api_key = st.sidebar.text_input("Groq API Key", type="password", value=os.getenv("GROQ_API_KEY", ""))
if api_key:
    os.environ["GROQ_API_KEY"] = api_key
    if not api_key.startswith("gsk_"): # Groq keys typically start with 'gsk_'
        st.sidebar.warning("⚠️ Groq API key might have an unusual format. It typically starts with 'gsk_'.")
else:
    st.sidebar.error("⚠️ Please enter your Groq API key")

st.sidebar.markdown("---")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
# Renaming the file path to reflect the change to Groq/HuggingFace, though not strictly necessary
file_path = "faiss_store_groq_hf.pkl" 

# Optional debug: show recent log contents
show_logs = st.sidebar.checkbox("Show logs")

main_placeholder = st.empty()

if process_url_clicked:
    # Check if API key is set before proceeding
    if not os.getenv("GROQ_API_KEY"):
        st.error("Please set your Groq API Key in the sidebar.")
        st.stop()
        
    # load data
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = []
    # filter out empty URL inputs
    target_urls = [u for u in urls if u and u.strip()]
    
    if not target_urls:
        st.error("Please provide at least one URL in the sidebar.")
    else:
        # --- REPLACEMENT FOR SeleniumURLLoader using requests + BeautifulSoup ---
        loaded_docs = []
        for u in target_urls:
            doc = fetch_and_parse_url(u)
            if doc:
                loaded_docs.append(doc)
        
        data = loaded_docs
        logger.info("Total documents loaded after fetching and parsing: %d", len(data))
        
        if not data:
            st.error("Could not successfully load content from any provided URLs.")
            st.stop()
        # --- END OF REPLACEMENT ---
    
        # split data
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000
        )
        main_placeholder.text("Text Splitter...Started...✅✅✅")
        docs = text_splitter.split_documents(data)

        # create embeddings and save it to FAISS index (using create_embeddings)
        try:
            vectorstore = create_embeddings(docs)
            main_placeholder.text("Embedding Vector Started Building...✅✅✅")
            time.sleep(2)
        except Exception as e:
            st.error("Failed to create embeddings. Please check the logs for details.")
            logger.exception("Error creating embeddings")
            if show_logs:
                st.error(f"Error details: {str(e)}")
            st.stop()

        # Save the FAISS index to a pickle file
        with open(file_path, "wb") as f:
            pickle.dump(vectorstore, f)

    if show_logs:
        try:
            with open(LOG_PATH, "r", encoding="utf-8") as lf:
                logs = lf.read().splitlines()
            # show last 200 lines to avoid flooding UI
            st.subheader("Recent logs")
            st.text("\n".join(logs[-200:]))
        except Exception:
            st.warning("Could not read log file.")

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        # Check if API key is set before proceeding
        if not os.getenv("GROQ_API_KEY"):
            st.error("Please set your Groq API Key in the sidebar.")
            st.stop()
            
        with open(file_path, "rb") as f:
            try:
                vectorstore = pickle.load(f)
                result = get_llm_response(vectorstore, query)
            except Exception as e:
                # Log a general exception for processing query if the specific Groq errors weren't caught
                st.error("Failed to process query. Please check the logs for details.")
                logger.exception("Error processing query")
                if show_logs:
                    st.error(f"Error details: {str(e)}")
                st.stop()
                
            # result will be a dictionary of this format --> {"answer": "", "sources": [] }
            st.header("Answer")
            st.write(result["answer"])

            # Display sources, if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")  # Split the sources by newline
                for source in sources_list:
                    # Only display non-empty source entries
                    if source.strip():

                        st.write(source.strip())
