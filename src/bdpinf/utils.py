"""
Common utilities for API data collection and processing.
"""

import os
import urllib.request


def ensure_directory(path: str) -> None:
    """Ensure a directory exists, creating it if necessary."""
    os.makedirs(path, exist_ok=True)


def download_file(url: str, local_path: str, overwrite: bool = False) -> None:
    """
    Download a file from a URL to a local path.

    Args:
        url: The URL to download from
        local_path: The local path to save to
        overwrite: Whether to overwrite existing files

    Raises:
        Exception: If download fails
    """
    if not overwrite and os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return

    try:
        print(f"Downloading {url} to {local_path}")
        # Ensure directory exists
        ensure_directory(os.path.dirname(local_path))
        urllib.request.urlretrieve(url, local_path)
    except Exception as e:
        raise Exception(f"Error downloading {url}: {e}")
