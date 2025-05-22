import requests
from bs4 import BeautifulSoup
import time
import random
from tabulate import tabulate

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) ' +
                  'AppleWebKit/537.36 (KHTML, like Gecko) ' +
                  'Chrome/114.0.0.0 Safari/537.36'
}

TICKERS = ['AAPL', 'GOOGL', 'MSFT', 'NVDA', 'TSLA']

def get_stock_info(ticker):
    url = f"https://finance.yahoo.com/quote/{ticker}"
    try:
        response = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(response.text, 'html.parser')

        price_tag = soup.find('fin-streamer', {'data-field': 'regularMarketPrice'})
        change_tag = soup.find('fin-streamer', {'data-field': 'regularMarketChange'})
        percent_tag = soup.find('fin-streamer', {'data-field': 'regularMarketChangePercent'})

        return {
            'Ticker': ticker,
            'Price': price_tag.text if price_tag else 'N/A',
            'Change': change_tag.text if change_tag else 'N/A',
            '% Change': percent_tag.text if percent_tag else 'N/A'
        }
    except Exception as e:
        return {'Ticker': ticker, 'Price': 'Error', 'Change': 'Error', '% Change': 'Error'}

def main():
    print("Fetching NASDAQ stock data...\n")
    all_data = []
    for ticker in TICKERS:
        stock_data = get_stock_info(ticker)
        all_data.append(stock_data)
        time.sleep(random.uniform(1, 3))  # polite scraping

    print(tabulate(all_data, headers="keys", tablefmt="fancy_grid"))

if __name__ == "__main__":
    main()
