import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from yt_dlp import YoutubeDL


st.set_page_config(page_title="Langchain: Summarize Text from YT or Website")
st.title("Summarize Text from YouTube or Website using Langchain")
st.subheader("Summarize URL")

with st.sidebar:
    groq_api_key = st.text_input("GROQ API KEY", type="password", value="")
    model_selection = st.selectbox(
        "Select model",
        [
            "llama-3.1-70b-versatile",
            "llama-3.1-8b-instant",
            "gemma-7b-it",
            "llama-3.2-90b-vision-preview",
            "llama3-70b-8192",
            "mixtral-8x7b-32768",
        ],
    )


llm = ChatGroq(model=model_selection, groq_api_key=groq_api_key)
prompt_template = """
Provide a summary of the following content in 300 words:
content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

url = st.text_input("Enter a URL (YouTube or website):", label_visibility="collapsed")

def fetch_youtube_data(url):
    with YoutubeDL({'format': 'bestaudio'}) as ydl:
        info = ydl.extract_info(url, download=False)
    return info.get("title", ""), info.get("description", ""), info.get("url", "")


if st.button("Summarize"):
    if not groq_api_key or not url.strip():
        st.error("Please enter both the GROQ API KEY and a valid URL.")
    elif not validators.url(url):
        st.error("Invalid URL. Please enter a valid one.")
    else:
        try:
            with st.spinner("Processing..."):
                if "youtube.com" in url or "youtu.be" in url:
                    try:
                        title, description, _ = fetch_youtube_data(url)
                        st.write(f"Video Title: {title}")
                        st.write(f"Video Description: {description}")
                        loader = YoutubeLoader.from_youtube_url(url, add_video_info=False)
                    except Exception as yt_err:
                        st.error("Failed to fetch YouTube video details.")
                        st.exception(str(yt_err))
                        loader = None
                else:
                    loader = UnstructuredURLLoader(
                        urls=[url],
                        ssl_verify=False,
                        headers={
                            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"
                        },
                    )

                if loader:
                    docs = loader.load()
                    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                    output = chain.run(docs)
                    st.success("Summary:")
                    st.write(output)

        except Exception as e:
            st.error("An error occurred while processing the request.")
            st.exception(f"Details: {str(e)}")
