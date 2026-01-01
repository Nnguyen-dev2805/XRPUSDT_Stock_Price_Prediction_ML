import requests
import pandas as pd
import time
from datetime import datetime

URL = "https://fapi.binance.com/fapi/v1/klines"

symbol = "XRPUSDT"
interval = "1d"
limit = 1500

def to_ms(date_str):
    return int(datetime.strptime(date_str, "%Y-%m-%d").timestamp() * 1000)

start_time = to_ms("2024-05-27")
end_time   = to_ms("2024-12-30")

rows = []
print("Dang chay")
while start_time < end_time:
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_time,
        "endTime": end_time,
        "limit": limit
    }

    data = requests.get(URL, params=params).json()
    if not data:
        break

    for k in data:
        rows.append([
            datetime.utcfromtimestamp(k[0]/1000).strftime("%Y-%m-%d"),
            float(k[1]),  # Open
            float(k[2]),  # High
            float(k[3]),  # Low
            float(k[4]),  # Price = Close
            float(k[5])   # Vol
        ])

    start_time = data[-1][0] + 1
    time.sleep(0.2)

df = pd.DataFrame(rows, columns=[
    "Date", "Open", "High", "Low", "Price", "Vol"
])

df.to_csv("future.csv", index=False)
print("DONE ✔")
