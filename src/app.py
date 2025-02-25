from typing import List, Dict, Optional
from newsdataapi import NewsDataApiClient
import os
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from time import sleep
import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="News Headline Generator",
    page_icon="📰",
    layout="wide"
)
# Load environment variables
load_dotenv()
api_key = os.getenv("NEWS_API_KEY")
if not api_key:
    raise ValueError("NEWS_API_KEY not found in environment variables")

# Initialize API client
api = NewsDataApiClient(apikey=api_key)

def get_news_title() -> str:
    """
    Get a list of news titles with error handling and rate limiting.


    Returns:
        str: A string representing a list of news titles

    Raises:
        Exception: If API call fails
    """
    try:
        # Add rate limiting
        sleep(0.1)  # 100ms delay between requests
        
        response = api.latest_api(
           language= "en",
            removeduplicate=True
        )
        
        if not response.get('results'):
            return []
            
        return response['results'][0]['title']
        
    except Exception as e:
        print(f"Error fetching news: {str(e)}")
        return []


st.write(f"News API response:       {get_news_title()}")
# Create news agent 
myagent = Agent(
    name="News Headline Generator",
    tools=[get_news_title],
    description="""You are a news headline generator that:
    1. Fetches latest news headlines from {get_news_title()} this will give title with extra info like source name but you only have to bring headline
    2. Generates one opinion questions realted to the headline
    3. Provides outcome option of this opinion
    your aim to spark meaningful conversations about current events.""",
    markdown=True,
    show_tool_calls=True
)


run_result =myagent.run("Generate opinion questions for the latest news")
st.markdown("-------------------------LLM Result -------------------------")
st.markdown(run_result.content)