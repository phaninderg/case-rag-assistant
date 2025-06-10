import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import hashlib
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def generate_id(prefix: str = "") -> str:
    """Generate a unique ID with an optional prefix."""
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    random_str = hashlib.md5(timestamp.encode()).hexdigest()[:8]
    return f"{prefix}_{random_str}" if prefix else random_str

def save_json(data: Any, file_path: Union[str, Path]) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to be saved as JSON
        file_path: Path to the file (can be string or Path object)
    """
    # Convert to Path object if it's a string
    path = Path(file_path) if isinstance(file_path, str) else file_path
    
    # Create parent directories if they don't exist
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the data to the file
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_json(file_path: Union[str, Path]) -> Any:
    """
    Load data from a JSON file.
    
    Args:
        file_path: Path to the JSON file (can be string or Path object)
        
    Returns:
        The loaded data, or None if the file doesn't exist
    """
    # Convert to Path object if it's a string
    path = Path(file_path) if isinstance(file_path, str) else file_path
    
    if not path.exists():
        return None
        
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Error loading JSON from {path}: {str(e)}")
        return None

def format_timestamp(timestamp: Optional[Union[str, datetime]] = None) -> str:
    """
    Format a timestamp to ISO format string.
    
    Args:
        timestamp: Optional timestamp (string or datetime). If None, uses current UTC time.
        
    Returns:
        ISO formatted timestamp string
    """
    if timestamp is None:
        return datetime.utcnow().isoformat()
    elif isinstance(timestamp, datetime):
        return timestamp.isoformat()
    # If it's already a string, assume it's in the correct format
    return timestamp

def validate_tags(tags: Any) -> List[str]:
    """Validate and normalize tags."""
    if not tags:
        return []
    if isinstance(tags, str):
        return [tag.strip() for tag in tags.split(",") if tag.strip()]
    if isinstance(tags, (list, tuple, set)):
        return [str(tag).strip() for tag in tags if str(tag).strip()]
    return []
