"""Download IBM's public Telco benchmark without personal Drive credentials."""
from pathlib import Path
from urllib.request import urlopen
import hashlib

URL = 'https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv'
PATH = Path(__file__).parent / 'data' / 'telco.csv'

def download():
    PATH.parent.mkdir(exist_ok=True)
    if not PATH.exists():
        with urlopen(URL, timeout=60) as response:
            PATH.write_bytes(response.read())
    return PATH

if __name__ == '__main__':
    path = download()
    print(path, hashlib.sha256(path.read_bytes()).hexdigest())
