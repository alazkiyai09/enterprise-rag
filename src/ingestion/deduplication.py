"""Document deduplication helpers extracted into the unified layout."""

from hashlib import sha256


def content_hash(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def is_duplicate(text: str, seen_hashes: set[str]) -> bool:
    digest = content_hash(text)
    if digest in seen_hashes:
        return True
    seen_hashes.add(digest)
    return False

