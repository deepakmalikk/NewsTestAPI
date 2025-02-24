from newsdataapi import NewsDataApiClient
import os 
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("API_KEY")
#initialize API client
api = NewsDataApiClient(apikey=api_key)

#To get latest news related to country india and language english and removing duplicate news articles 
response = api.latest_api(country="in",language="en", removeduplicate=True)
news_title = [i['title'] for i in response['results']]
print(news_title)
