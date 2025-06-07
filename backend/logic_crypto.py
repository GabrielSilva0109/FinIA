import requests
import pandas as pd

class CryptoAnalyzer:
    def __init__(self):
        self.base_url = "https://api.coingecko.com/api/v3"

    def get_price_data(self, coin_id, days=30):
        url = f"{self.base_url}/coins/{coin_id}/market_chart"
        params = {"vs_currency": "usd", "days": days}
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            prices = data.get('prices', [])
            if not prices:
                raise ValueError("No price data found.")
            df = pd.DataFrame(prices, columns=['timestamp', 'price'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except (requests.RequestException, ValueError) as e:
            print(f"Error fetching price data: {e}")
            return pd.DataFrame(columns=['price'])

    def calculate_moving_average(self, price_data, window=7):
        if price_data.empty:
            return pd.Series(dtype=float)
        return price_data['price'].rolling(window=window, min_periods=1).mean()

    def calculate_volatility(self, price_data):
        if price_data.empty:
            return float('nan')
        return price_data['price'].pct_change().std()

    def analyze(self, coin_id, days=30, ma_window=7):
        price_data = self.get_price_data(coin_id, days)
        moving_average = self.calculate_moving_average(price_data, window=ma_window)
        volatility = self.calculate_volatility(price_data)
        latest_price = price_data['price'].iloc[-1] if not price_data.empty else float('nan')

        return {
            'moving_average': moving_average,
            'volatility': volatility,
            'latest_price': latest_price
        }
