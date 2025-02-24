from newsdataapi import NewsDataAPIClient
import os 
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("API_KEY")
#initialize API client
news = NewsDataAPIClient(apikey=api_key)