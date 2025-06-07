import requests
import pandas as pd

class CryptoAnalyzer:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.coingecko.com/api/v3"

    def get_price_data(self, coin_id, days=30):
        url = f"{self.base_url}/coins/{coin_id}/market_chart?vs_currency=usd&days={days}"
        response = requests.get(url)
        data = response.json()
        prices = data['prices']
        return pd.DataFrame(prices, columns=['timestamp', 'price']).set_index('timestamp')

    def calculate_moving_average(self, price_data, window=7):
        return price_data['price'].rolling(window=window).mean()

    def calculate_volatility(self, price_data):
        return price_data['price'].pct_change().std()

    def analyze(self, coin_id):
        price_data = self.get_price_data(coin_id)
        moving_average = self.calculate_moving_average(price_data)
        volatility = self.calculate_volatility(price_data)

        return {
            'moving_average': moving_average,
            'volatility': volatility,
            'latest_price': price_data['price'].iloc[-1]
        }
