#!/usr/bin/env python3
"""
Script to import cases from a website using the Case RAG Assistant API.

Example usage:
    python scripts/import_cases.py https://example.com/cases 10
"""

import sys
import requests
import json
import time
from urllib.parse import urljoin

def main():
    if len(sys.argv) < 2:
        print("Usage: python import_cases.py <base_url> [max_pages]")
        sys.exit(1)
    
    base_url = sys.argv[1]
    max_pages = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    # Start the import
    response = requests.post(
        "http://localhost:8000/api/import/cases",
        json={"url": base_url, "max_pages": max_pages}
    )
    
    if response.status_code != 202:
        print(f"Error starting import: {response.status_code} - {response.text}")
        sys.exit(1)
    
    task = response.json()
    task_id = task["task_id"]
    print(f"Import started. Task ID: {task_id}")
    print("Checking status...")
    
    # Poll for status
    while True:
        status_resp = requests.get(f"http://localhost:8000/api/import/status/{task_id}")
        if status_resp.status_code != 200:
            print(f"Error getting status: {status_resp.status_code} - {status_resp.text}")
            break
            
        status_data = status_resp.json()
        status = status_data.get("status")
        
        if status == "completed":
            result = status_data.get("result", {})
            print(f"\nImport completed!")
            print(f"Pages visited: {result.get('total_visited', 0)}")
            print(f"Cases imported: {result.get('imported_count', 0)}")
            break
        elif status == "failed":
            print(f"\nImport failed: {status_data.get('error', 'Unknown error')}")
            break
        else:
            print(f"Status: {status}...", end="\r")
            time.sleep(2)

if __name__ == "__main__":
    main()
