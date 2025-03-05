from typing import Optional
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

def api_setup() -> dict:
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
        Returns an empty string if no headline is found or if an error occurs.
    """
    try:
        sleep(0.1)  # Rate limiting to prevent API overuse
        api_keys = api_setup()
        news_api = NewsDataApiClient(apikey=api_keys.get("News"))

        response = news_api.latest_api(language="en", removeduplicate=True)

        if not response.get('results'):
            return ""
        return response['results'][0]['title']
    
    except Exception as e:
        st.error(f"Error fetching news: {str(e)}")
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
            "OpenAIChat": ["gpt-4o", "gpt-4o-mini", "gpt-4.5-preview"],
            "xAI": ["grok-2-1212", "grok-beta"],
            "Claude": ["claude-3-7-sonnet-20250219", "claude-3-5-haiku-20241022","claude-3-opus-20240229"],
            "Gemini": ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-flash-8b"]
        }

        provider_options = ["Select an LLM Provider"] + list(llm_models.keys())
        selected_llm = st.selectbox("Choose LLM Provider:", provider_options)

        if selected_llm == "Select an LLM Provider":
            st.write("Please select a valid LLM provider from the dropdown.")
            return None
        
        llm_options = ["Select a Model"] + llm_models[selected_llm]
        selected_model = st.selectbox("Choose Model:", llm_options)

        if selected_model == "Select a Model":
            st.write("Please select a valid model from the dropdown.")
            return None

        # Validate API keys
        try:
            api_keys = api_setup()
            _ = api_keys.get(selected_llm)
        except ValueError as e:
            st.error(str(e))
            return None

        st.write(f"**Selected LLM Provider:** {selected_llm}")
        st.write(f"**Selected Model:** {selected_model}")

        return selected_llm, selected_model

# --------------------- Initialize Selected LLM ---------------------

def get_model(selection: Optional[tuple]):
    """
    Returns an instance of the selected LLM model using the chosen provider.

    Args:
        selection (Optional[tuple]): A tuple of (LLM provider, model) from the sidebar.

    Returns:
        An instance of the selected LLM class with its API key.
    """
    if selection is None:
        return None

    llm_provider, selected_model = selection
    api_keys = api_setup()

    if llm_provider == "OpenAIChat":
        return OpenAIChat(id=selected_model, api_key=api_keys["OpenAI"])
    elif llm_provider == "xAI":
        return xAI(id=selected_model, api_key=api_keys["xAI"])
    elif llm_provider == "Claude":
        return Claude(id=selected_model, api_key=api_keys["Claude"])
    elif llm_provider == "Gemini":
        return Gemini(id=selected_model, api_key=api_keys["Gemini"])
    else:
        raise ValueError(f"Unsupported LLM provider: {llm_provider}")

# --------------------- User Input for News Headline ---------------------

def get_user_input() -> str:
    """
    Displays the latest news headline for reference and provides a text input
    for the user to paste the headline they wish to test.

    Returns:
        str: The user's input text (expected to be a news headline).
    """
    news_headline = get_news_title()
    if news_headline:
        st.markdown(f"**News API Headline:** {news_headline}")
        st.info("Copy the headline above and paste it into the box below to test with different LLMs.")
    else:
        st.warning("No news headline available. Please check the API key or try again later.")
    
    user_query = st.text_input("Paste your news headline here:")
    return user_query

# --------------------- News Headline Generation Agent ---------------------

def main_Agent(user_query: str, selection: Optional[tuple]):
    """
    Creates and runs the News Headline Generator agent.
    
    The agent:
      - Takes the provided user query (which includes a news title and extra info) 
        and extracts only the headline.
      - Generates one opinion question related to the headline.
      - Provides possible opinion outcomes to spark discussions.
    """
    if not user_query:
        st.info("Please enter or paste a news headline to proceed.")
        return

    if selection is None:
        st.error("No LLM model selected. Please choose an option from the sidebar.")
        return

    llm_model = get_model(selection)
    if llm_model is None:
        st.error("Error initializing the LLM model.")
        return

    myagent = Agent(
        name="News Headline Generator",
        model=llm_model,
        description=(
           """You are a precision news prediction generator. Strictly follow these rules:

                1. HEADLINE PROCESSING:
                - Extract ONLY the core headline text, removing source names/extra info
                - Identify: [ENTITY], [NUMERIC_VALUE], [TIMEFRAME], [EVENT_DATE]

                2. QUESTION GENERATION:
                - Format: "Will [ENTITY] [ACTION] [CRITERIA] [TIMEFRAME]?"
                - Use exact numbers/dates from headline
                - Timeframe must match: today/this week/YYYY-MM-DD

                3. OPTIONS:
                - 4 mutually exclusive options
                - Market questions: Bullish/Bearish/Neutral/Alternative
                - Policy questions: Yes/No/Partial/Alternative

                4. VALIDATION:
                - Numbers must match headline exactly
                - One clear resolution criteria
                - No hypothetical scenarios

                OUTPUT FORMAT:
                {
                "headline": "Original Headline",
                "question": "Will...?",
                "date_pattern": "timeframe",
                "category": "CATEGORY",
                "source": "Cleaned Source",
                "options": [
                    {"id": "A", "text": "Option1"},
                    {"id": "B", "text": "Option2"},
                    {"id": "C", "text": "Option3"},
                    {"id": "D", "text": "Option4"}
                ]
                }

                EXAMPLE:
                Headline: "Fed hints at June rate pause"
                →
                {
                "headline": "Fed hints at June rate pause",
                "question": "Will the Fed pause rate hikes at the June 13-14 meeting?",
                "date_pattern": "2023-06-14",
                "category": "Financials",
                "source": "Bloomberg",
                "options": [
                    {"id": "A", "text": "Yes, full pause (0bps)"},
                    {"id": "B", "text": "No, 25bps hike"},
                    {"id": "C", "text": "No, 50bps hike"},
                    {"id": "D", "text": "Mixed outcome"}
                ]
                }"""
        ),
        markdown=True,
        show_tool_calls=True
    )

    run_result = myagent.run(user_query)
    st.markdown("------------------------- LLM Result -------------------------")
    st.markdown(run_result.content)

# --------------------- Main Execution ---------------------

def main():
    page_setup()
    # Call the LLM selector so the sidebar is always rendered
    selection = llm_selector()
    user_query = get_user_input()
    main_Agent(user_query, selection)

if __name__ == "__main__":
    main()