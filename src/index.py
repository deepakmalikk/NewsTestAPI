import requests
import os
from dotenv import load_dotenv

load_dotenv()   

API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise ValueError("Missing API key")
else:
    print("API Key found")

url ="https://newsdata.io/api/1/latest?apikey={}".format(API_KEY)

response = requests.get(url)
result= response.json()
for i in result['results']:
    print(i['title'])