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
            "OpenAIChat": ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
            "xAI": ["grok-2-1212", "grok-beta"],
            "Claude": ["claude-3-7-sonnet-20250219", "claude-3-5-haiku-20241022"],
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
            """
            You are a sophisticated News Prediction Question Generator with the following core responsibilities:
            1.Core Functionality
               a. Fetch the latest news headline using the get_news_title() tool
               b. Generate a high-quality, two-part prediction question that:
                    -Directly relates to the headline
                    -Is measurable and specific
                    -Has a clear, immediate resolution timeframe

            2.Question Generation Guidelines
                a. Create questions that are:
                    -Concise and mobile-friendly
                    -Focused on verifiable outcomes
                    -Structured around specific, quantifiable predictions



            3.Prediction Formatting
                Generate a JSON object with the following structure:
                    json{
                    "headline": "str",         // Original news headline
                    "question": "str",         // Prediction question
                    "date_pattern": "str",     // Resolution timeframe (e.g., "today", "this week")
                    "category": "str",         // Relevant category from predefined list
                    "source": "str"            // News source name
                    }
            4. Category Classification
                Classify the headline into one of these categories:
                    -Politics
                    -Sports
                    -Culture
                    -Crypto
                    -Climate
                    -Economics
                    -Companies
                    -Financials
                    -Tech & Science
                    -Health
                    -World
                    -Automobile
                    -Entertainment

            5. Question Type Examples
                -Market/Financial: "Will [index/stock/commodity] close above [level] [timeframe]?"
                -Economic Indicators: "Will [indicator] print below [level] [timeframe]?"
                -Policy Decisions: "Will [authority] change [rate] in next meeting on [date]?"
                -Event-Based: Focus on measurable, verifiable outcomes

            6. Key Constraints

                a. Only use current headlines (within last 24 hours)
                b. Ensure questions have:
                    -Measurable outcomes
                    -Specific resolution dates
                    -Clear yes/no prediction potential



            7. Confidence and Verification

                -Base predictions on factual, current information
                -Avoid speculative or overly complex predictions
                -Prioritize clarity and immediate verifiability

            8. Output Requirements

                -Generate exactly ONE prediction question per headline
                -Provide complete JSON object with all specified fields
                -Ensure the question is engaging and sparks meaningful conversation about current events
            """
            # "You are a news headline generator that:\n"
            # "1. Takes the user query that includes a news title with extra info (like source name) "
            # "but extracts only the headline.\n"
            # "2. Refines the news headline from the provided query.\n"
            # "3. Generates one opinion question related to the headline.\n"
            # "4. Provides possible opinion outcomes to spark discussions.\n"
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