
import requests
import pandas as pd
from io import StringIO
from bs4 import BeautifulSoup

url = "https://www.sports-reference.com/cbb/seasons/2025-ratings.html"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.content, 'html.parser')
table = soup.find('table', {'id': 'ratings'})
df = pd.read_html(StringIO(str(table)))[0]

print("Columns:", df.columns)
print("First 5 rows:")
print(df.head(5))
