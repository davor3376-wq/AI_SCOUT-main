import os
import httpx
from dotenv import load_dotenv

load_dotenv()
JULES_KEY = os.getenv("JULES_API_KEY")

def get_sources():
    headers = {"x-goog-api-key": JULES_KEY}
    url = "https://jules.googleapis.com/v1alpha/sources"
    
    with httpx.Client() as client:
        response = client.get(url, headers=headers)
        if response.status_code == 200:
            sources = response.json().get("sources", [])
            if not sources:
                print("No sources found. Is the Jules GitHub App installed on the repo?")
            for s in sources:
                print(f"REPO: {s['githubRepo']['repo']}")
                print(f"SOURCE_NAME: {s['name']}\n")
        else:
            print(f"Error {response.status_code}: {response.text}")

if __name__ == "__main__":
    get_sources()