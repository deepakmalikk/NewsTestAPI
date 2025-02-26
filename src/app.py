from typing import List, Dict, Optional
from newsdataapi import NewsDataApiClient
import os
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.models.anthropic import Claude
from agno.models.google import Gemini
from agno.models.xai import xAI
from time import sleep
import streamlit as st


# --------------------- Page Setup ---------------------

def page_setup():
    """
    Configures the Streamlit page settings, such as title, icon, and layout.
    """
    st.set_page_config(
        page_title="News Headline Generator",
        page_icon="📰",
        layout="wide"
    )


# --------------------- API Setup ---------------------

def api_setup():
    """
    Loads API keys from environment variables and validates them.
    
    Returns:
        dict: A dictionary containing API keys for various services.
    
    Raises:
        ValueError: If any API key is missing.
    """
    load_dotenv()
    api_keys = {
        "OpenAI": os.getenv("OPENAI_API_KEY"),
        "xAI": os.getenv("XAI_API_KEY"),
        "Claude": os.getenv("ANTHROPIC_API_KEY"),
        "Gemini": os.getenv("GOOGLE_API_KEY"),
        "News": os.getenv("NEWS_API_KEY")
    }
    
    for provider, key in api_keys.items():
        if not key:
            raise ValueError(f"API key for {provider} not found in environment variables")
    
    return api_keys


# --------------------- Fetch News Headlines ---------------------

def get_news_title() -> str:
    """
    Fetches the latest news headlines from the NewsData API.

    Returns:
        str: The latest news headline.

    If an error occurs or no headlines are found, it returns an empty string.
    """
    try:
        sleep(0.1)  # Rate limiting to prevent API overuse
        api = api_setup()
        news_api = NewsDataApiClient(apikey=api.get("News"))

        response = news_api.latest_api(language="en", removeduplicate=True)

        if not response.get('results'):
            return ""

        return response['results'][0]['title']
    
    except Exception as e:
        print(f"Error fetching news: {str(e)}")
        return ""


# --------------------- LLM Selection UI ---------------------

def llm_selector() -> Optional[tuple]:
    """
    Provides a Streamlit sidebar for selecting an LLM provider and model.

    Returns:
        tuple: (Selected LLM provider, Selected model) if valid choices are made, else None.
    """
    with st.sidebar:
        st.title("Select an LLM Model")

        # Available LLMs and their models
        llm_models = {
            "OpenAIChat": ["gpt-4o", "gpt-4o-mini","gpt-3.5-turbo"],
            "xAI": ["grok-2-1212", "grok-beta"],
            "Claude": ["claude-3-7-sonnet-20250219", "claude-3-5-haiku-20241022"],
            "Gemini": ["gemini-2.0-flash", "gemini-1.5-flash","gemini-1.5-flash-8b"]
        }

        provider_options = ["Select an LLM Provider"] + list(llm_models.keys())
        selected_llm = st.selectbox("Choose LLM Provider:", provider_options)

        if selected_llm == "Select an LLM Provider":
            st.write("Please select a valid LLM provider from the dropdown.")
            return None
        
        selected_model = st.selectbox("Choose Model:", llm_models[selected_llm])

        # Load API keys and handle missing keys
        try:
            api_keys = api_setup()
            selected_api_key = api_keys.get(selected_llm, None)
        except ValueError as e:
            st.error(str(e))
            return None

        # Display selected model details
        st.write(f"**Selected LLM Provider:** {selected_llm}")
        st.write(f"**Selected Model:** {selected_model}")

        return selected_llm, selected_model


# --------------------- Initialize Selected LLM ---------------------

def model():
    """
    Returns the selected LLM model object.

    Returns:
        An instance of the selected LLM class.

    Raises:
        ValueError: If an unsupported LLM provider is selected.
    """
    selection = llm_selector()
    if selection is None:
        return None

    llm, model = selection

    if llm == "OpenAIChat":
        return OpenAIChat(id=model)
    elif llm == "xAI":
        return xAI(id=model)
    elif llm == "Claude":
        return Claude(id=model)
    elif llm == "Gemini":
        return Gemini(id=model)
    else:
        raise ValueError(f"Unsupported LLM provider: {llm}")


# --------------------- News Headline Generation Agent ---------------------

def main_Agent():
    """
    Creates and runs the News Headline Generator agent.
    
    - Fetches the latest news headlines.
    - Generates opinion questions based on the headline.
    - Provides outcome options to spark discussions.
    """
    llm_model = model()
    if llm_model is None:
        st.error("No LLM model selected. Please choose an option from the sidebar.")
        return

    myagent = Agent(
        name="News Headline Generator",
        tools=[get_news_title],
        model=llm_model,
        description="""You are a news headline generator that:
        1. Fetches the latest news headlines from {get_news_title()}.
        2. Extracts only the headline (excluding source info).
        3. Generates one opinion question related to the headline.
        4. Provides possible opinion outcomes.
        Your goal is to encourage meaningful discussions on current events.""",
        markdown=True,
        show_tool_calls=True
    )

    run_result = myagent.run("Generate opinion questions for the latest news")

    st.markdown("-------------------------LLM Result -------------------------")
    st.markdown(run_result.content)


# --------------------- Main Execution ---------------------

if __name__ == "__main__":
    page_setup()
    main_Agent()
