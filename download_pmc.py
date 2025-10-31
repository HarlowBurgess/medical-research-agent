# download_pmc.py
import requests
import boto3
import time
import random
from pathlib import Path
from requests.exceptions import RequestException

s3_client = boto3.client('s3')


def retry_with_backoff(fn, max_attempts=4, initial_delay=1.0, backoff_factor=2.0, max_delay=30.0, jitter=0.1):
    """Call fn() and retry on exceptions or falsy/unsuccessful results.

    fn should be a zero-arg callable. If it raises an exception the call will
    be retried. If it returns a requests.Response the function will retry on
    status codes >= 500 (server errors). For other return types, falsy
    results will trigger a retry.
    """
    attempt = 0
    delay = initial_delay
    while True:
        attempt += 1
        try:
            result = fn()

            # If it's a requests.Response, consider server errors retriable
            if isinstance(result, requests.Response):
                if 500 <= result.status_code:
                    raise RequestException(f"Server error: {result.status_code}")
                return result

            # For other types, consider falsy values retriable
            if result:
                return result
            raise RequestException("Received falsy result, retrying")

        except Exception as exc:
            if attempt >= max_attempts:
                raise

            # sleep with exponential backoff + jitter
            jitter_val = random.uniform(-jitter * delay, jitter * delay)
            sleep_for = min(max_delay, delay + jitter_val)
            time.sleep(max(0.0, sleep_for))
            delay *= backoff_factor


def download_pmc_subset():
    """Download curated medical papers from PMC OA with retries on network calls."""

    mesh_terms = [
        # "Neoplasms",
        # "Cardiovascular Diseases",
        "Diabetes Mellitus",
        "Neurodegenerative Diseases",
        "Infectious Disease",
    ]

    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

    for mesh_term in mesh_terms:
        search_url = f"{base_url}esearch.fcgi"
        params = {
            "db": "pmc",
            "term": f"{mesh_term}[MeSH] AND open access[filter]",
            "retmax": 2000,
            "retmode": "json",
            "datetype": "pdat",
            "mindate": "2020",
            "maxdate": "2024",
        }

        def do_search():
            return requests.get(search_url, params=params, timeout=30)

        response = retry_with_backoff(do_search)
        pmcids = response.json()["esearchresult"].get("idlist", [])

        for pmcid in pmcids:
            fetch_url = f"{base_url}efetch.fcgi"
            params = {"db": "pmc", "id": pmcid, "rettype": "xml"}

            def do_fetch():
                return requests.get(fetch_url, params=params, timeout=30)

            fetch_resp = retry_with_backoff(do_fetch)
            paper_xml = fetch_resp.text

            def do_put():
                return s3_client.put_object(
                    Bucket="medical-research-agent-documents",
                    Key=f"raw/{mesh_term}/{pmcid}.xml",
                    Body=paper_xml,
                )

            # boto3 returns a dict; treat falsy/exception as retriable
            retry_with_backoff(do_put)


if __name__ == "__main__":
    download_pmc_subset()