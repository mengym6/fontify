"""Sealed original/target identity checks for strict Run B references."""

STRICT_REFERENCE_POLICY = "distinct-original-reference-v1"


def reference_identity(pair, source_hashes):
    """Require both image identities from the freshly verified cache seal."""
    paths = (pair["original_path"], pair["target_path"])
    values = tuple(source_hashes.get(path) for path in paths)
    if any(not isinstance(value, str) or len(value) != 64 for value in values):
        raise ValueError("Strict references require sealed original/target SHA256")
    return values


def references_distinct(query, reference, source_hashes):
    """Reject either an original-image alias or a repeated target image."""
    left = reference_identity(query, source_hashes)
    right = reference_identity(reference, source_hashes)
    return all(first != second for first, second in zip(left, right))
