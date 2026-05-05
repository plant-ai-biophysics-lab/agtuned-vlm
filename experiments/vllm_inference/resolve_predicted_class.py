def resolve_predicted_class(response: str, class_names: list[str]) -> str:
    """Map free-text response to the longest matching class name (case-insensitive)."""
    response_lower = response.lower()
    matches = [c for c in class_names if c.lower() in response_lower]
    if not matches:
        return "unknown"
    return max(matches, key=len)

response = "This is an image of a tomato plant with early or late blight disease."
class_names = ["early blight", "late blight", "leaf curl"]

matches = resolve_predicted_class(response, class_names)
print(matches)
