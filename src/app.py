from newsdataapi import NewsDataApiClient
import os 
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("API_KEY")
#initialize API client
api = NewsDataApiClient(apikey=api_key)

#To get latest news related to country india 
response = api.latest_api(country="au,us")
news_title = [i['title'] for i in response['results']]

print(len(news_title))
