from typing import Optional
import os
from time import sleep
import streamlit as st
from dotenv import load_dotenv
from newsdataapi import NewsDataApiClient
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.models.anthropic import Claude
from agno.models.google import Gemini
from agno.models.xai import xAI
from datetime import date

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

# --------------------- API Setup with Caching ---------------------

@st.cache_data(show_spinner=False)
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

# --------------------- Fetch News Headlines with Caching ---------------------

@st.cache_data(ttl=60, show_spinner=False)
def fetch_news_title() -> str:
    """
    Fetches the latest news headline from the NewsData API and caches it for 60 seconds.
    
    Returns:
        str: The latest news headline, or an empty string if none is found.
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

def get_news_title() -> str:
    """
    Wrapper to retrieve the cached news headline.
    """
    return fetch_news_title()

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
            "OpenAIChat": ["gpt-4o", "gpt-4.5-preview"],
            "xAI": ["grok-2-1212", "grok-beta"],
            "Claude": ["claude-3-7-sonnet-20250219", "claude-3-5-haiku-20241022", "claude-3-opus-20240229"],
            "Gemini": ["gemini-2.0-flash", "gemini-1.5-flash-8b"]
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
            _ = api_setup()
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
    
    return st.text_input("Paste your news headline here:")

# --------------------- News Headline Generation Agent ---------------------

def main_Agent(user_query: str, selection: Optional[tuple]):
    """
    Creates and runs the News Headline Generator agent.
    
    The agent:
      - Extracts the core headline text.
      - Generates an opinion question and options based on the headline.
      - Ensures that any generated "date_pattern" is in the future relative to the provided current date.
    
    Here we inject the current system date as a reference into the prompt.
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

    # Get today's date as reference
    current_date = date.today().strftime('%Y-%m-%d')
    
    myagent = Agent(
        name="News Headline Generator",
        model=llm_model,
        description=(
           f"""You are a precision news prediction generator. Strictly follow these rules:

1. HEADLINE PROCESSING:
   - Extract ONLY the core headline text, removing source names/extra info.
   - Identify key elements: [ENTITY], [NUMERIC_VALUE], [TIMEFRAME], [EVENT_DATE].

2. QUESTION GENERATION:
   - Format: "Will [ENTITY] [ACTION] [CRITERIA] [TIMEFRAME]?"
   - Use exact numbers/dates from the headline.
   - Timeframe must match the formats: today / this week / YYYY-MM-DD.

3. OPTIONS:
   - Generate 4 mutually exclusive options.
   - For market questions: options like Bullish/Bearish/Neutral/Alternative.
   - For policy questions: options like Yes/No/Partial/Alternative.

4. VALIDATION:
   - Ensure numbers match exactly those in the headline.
   - Use one clear resolution criteria.
   - Avoid hypothetical scenarios.

5. DATE PREDICTION RULE:
   - Based on the current system date ({current_date}), ensure that any date in "date_pattern" is in the future.
   - If the generated date is in the past relative to {current_date}, update it to the current or an upcoming year.

OUTPUT FORMAT:
{{
  "headline": "Original Headline",
  "question": "Will...?",
  "date_pattern": "timeframe",
  "category": "CATEGORY",
  "source": "Cleaned Source",
  "options": [
      {{"id": "A", "text": "Option1"}},
      {{"id": "B", "text": "Option2"}},
      {{"id": "C", "text": "Option3"}},
      {{"id": "D", "text": "Option4"}}
  ]
}}

EXAMPLE:
Headline: "Fed hints at June rate pause"
→
{{
  "headline": "Fed hints at June rate pause",
  "question": "Will the Fed pause rate hikes at the June 13-14 meeting?",
  "date_pattern": "2025-06-14",
  "category": "Financials",
  "source": "Bloomberg",
  "options": [
      {{"id": "A", "text": "Yes, full pause (0bps)"}},
      {{"id": "B", "text": "No, 25bps hike"}},
      {{"id": "C", "text": "No, 50bps hike"}},
      {{"id": "D", "text": "Mixed outcome"}}
  ]
}}"""
        ),
        markdown=True,
        show_tool_calls=True
    )

    with st.spinner("Generating news headline prediction..."):
        run_result = myagent.run(user_query)
    
    st.markdown("------------------------- LLM Result -------------------------")
    st.markdown(run_result.content)

# --------------------- Main Execution ---------------------

def main():
    page_setup()
    # Render the sidebar LLM selector
    selection = llm_selector()
    user_query = get_user_input()
    main_Agent(user_query, selection)

if __name__ == "__main__":
    main()
