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
    st.set_page_config(
        page_title="News Headline Generator",
        page_icon="📰",
        layout="wide"
    )

# --------------------- API Setup with Caching ---------------------

@st.cache_data(show_spinner=False)
def api_setup() -> dict:
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
    return fetch_news_title()

# --------------------- LLM Selection UI ---------------------

def llm_selector() -> Optional[tuple]:
    with st.sidebar:
        st.title("Select an LLM Model")
        llm_models = {
            "OpenAIChat": ["gpt-4o", "gpt-4o-mini", "gpt-4.5-preview"],
            "xAI": ["grok-2-1212", "grok-beta"],
            "Claude": ["claude-3-7-sonnet-20250219", "claude-3-5-haiku-20241022", "claude-3-opus-20240229"],
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
    news_headline = get_news_title()
    if news_headline:
        st.markdown(f"**News API Headline:** {news_headline}")
        st.info("Copy the headline above and paste it into the box below to test with different LLMs.")
    else:
        st.warning("No news headline available. Please check the API key or try again later.")
    return st.text_input("Paste your news headline here:")

# --------------------- News Headline Generation Agent ---------------------

def main_Agent(user_query: str, selection: Optional[tuple]):
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

    current_date = date.today().strftime('%Y-%m-%d')
    
    myagent = Agent(
        name="News Headline Generator",
        model=llm_model,
        description=(
           f"""You are a precision news prediction generator. Follow these strict rules:

1. HEADLINE PROCESSING:
   - Extract ONLY the core headline text, removing source names/extra info.
   - Identify key elements: [ENTITY], [NUMERIC_VALUE], [TIMEFRAME], [EVENT_DATE].
   - If the headline indicates a definitive event (e.g., retirement, a final decision), note this context.

2. QUESTION GENERATION:
   - Create a question that is contextually appropriate for the headline.
   - The question may begin with any interrogative word (e.g., "Will", "Is", "Why", "How", etc.) based on the headline’s context.
   - Ensure the generated question is logically consistent with the headline.
   - If the headline indicates a definitive outcome, the question should explore potential reversals, alternative scenarios, or implications.

3. OPTIONS:
   - Generate 4 mutually exclusive options.
   - For market-related questions: options such as Bullish/Bearish/Neutral/Alternative.
   - For event or policy-related questions: options such as Yes/No/Partial/Alternative.
   - Ensure the options are consistent with the headline context.

4. VALIDATION:
   - Ensure numbers and dates match exactly those in the headline.
   - Avoid hypothetical scenarios that contradict the headline’s facts.

5. DATE PREDICTION RULE:
   - Based on the current system date ({current_date}), ensure that any date in "date_pattern" is in the future.
   - If the generated date is in the past relative to {current_date}, update it to the current or an upcoming year.

OUTPUT FORMAT:
{{
  "headline": "Original Headline",
  "question": "Question generated based on headline context.",
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
Headline: "Steve Smith retires from ODI cricket after Champions Trophy exit"
→
{{
  "headline": "Steve Smith retires from ODI cricket after Champions Trophy exit",
  "question": "Is there any possibility that Steve Smith might reverse his retirement decision or take on a non-playing role in Australia's next ODI series?",
  "date_pattern": "2025-06-01",
  "category": "Sports",
  "source": "The Hindu",
  "options": [
      {{"id": "A", "text": "No, remains retired from ODIs"}},
      {{"id": "B", "text": "Yes, returns in a playing role"}},
      {{"id": "C", "text": "Joins in a non-playing advisory capacity"}},
      {{"id": "D", "text": "Returns only for specific tournaments"}}
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
    selection = llm_selector()
    user_query = get_user_input()
    main_Agent(user_query, selection)

if __name__ == "__main__":
    main()


