import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import hashlib
from datetime import datetime

def generate_id(prefix: str = "") -> str:
    """Generate a unique ID with an optional prefix."""
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    random_str = hashlib.md5(timestamp.encode()).hexdigest()[:8]
    return f"{prefix}_{random_str}" if prefix else random_str

def save_json(data: Any, file_path: Path) -> None:
    """Save data to a JSON file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_json(file_path: Path) -> Any:
    """Load data from a JSON file."""
    if not file_path.exists():
        return None
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def format_timestamp(timestamp: Optional[str] = None) -> str:
    """Format a timestamp string."""
    if timestamp is None:
        timestamp = datetime.utcnow().isoformat()
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
