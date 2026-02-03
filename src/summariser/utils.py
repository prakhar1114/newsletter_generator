from summariser.config import MARKDOWN_PATH

def fetch_markdown_from_id(file_id: str) -> str:
    with open(MARKDOWN_PATH / f"{file_id}.md", "r") as f:
        return f.read()
