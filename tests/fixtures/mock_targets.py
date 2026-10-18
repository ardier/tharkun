"""Fixture for SG-5 mock generation stress tests."""

import hashlib
import hmac
import os
import sqlite3
import zlib
from pathlib import Path

import requests


# --- Scenario 1: HTTP calls ---

def fetch_user(user_id: int) -> dict:
    resp = requests.get(f"https://api.example.com/users/{user_id}")
    resp.raise_for_status()
    return resp.json()


def post_data(endpoint: str, payload: dict) -> int:
    resp = requests.post(endpoint, json=payload)
    return resp.status_code


# --- Scenario 2: Filesystem ---

def read_config(path: str) -> str:
    with open(path, "r") as f:
        return f.read()


def check_and_read(filepath: str) -> str:
    if not os.path.exists(filepath):
        return ""
    return Path(filepath).read_text(encoding="utf-8")


# --- Scenario 3: Database ---

def get_user_count(db_path: str) -> int:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM users")
    count = cursor.fetchone()[0]
    conn.close()
    return count


# --- Scenario 4: Checksums/Hashes ---

def verify_checksum(data: bytes, expected: int) -> bool:
    return zlib.crc32(data) == expected


def compute_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


def sign_message(key: bytes, msg: bytes) -> str:
    mac = hmac.new(key, msg, hashlib.sha256)
    return mac.hexdigest()


# --- Scenario 5: Mixed HTTP + Filesystem ---

def download_and_cache(url: str, cache_path: str) -> dict:
    import json
    cached = Path(cache_path)
    if cached.exists():
        return json.loads(cached.read_text())
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    cached.write_text(json.dumps(data))
    return data


# --- Scenario 6: Nested helpers ---

def _fetch_raw(url: str) -> dict:
    return requests.get(url).json()


def get_enriched(user_id: int) -> dict:
    raw = _fetch_raw(f"https://api.example.com/users/{user_id}")
    raw["enriched"] = True
    return raw


# --- Scenario 7: Class with attribute-chain ---

class ApiClient:
    def __init__(self, base_url: str):
        self.session = requests.Session()
        self.base_url = base_url

    def get_resource(self, resource_id: int) -> dict:
        resp = self.session.get(f"{self.base_url}/resources/{resource_id}")
        resp.raise_for_status()
        return resp.json()


# --- Scenario 8: Function with no externals (should be skipped) ---

def pure_function(x: int, y: int) -> int:
    return x + y
