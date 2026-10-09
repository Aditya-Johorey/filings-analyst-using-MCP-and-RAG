import os
from dotenv import load_dotenv
from sec_edgar_downloader import Downloader

load_dotenv()
dl = Downloader(os.environ["SEC_NAME"], os.environ["SEC_EMAIL"], "data/")

for ticker in ["AAPL", "MSFT", "TSLA"]:
    dl.get("10-K", ticker, limit = 2, download_details = True)
