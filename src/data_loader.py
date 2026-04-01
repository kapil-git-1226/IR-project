import os
import re
import json
from bs4 import BeautifulSoup
from typing import List, Dict


def parse_sgm_file(file_path: str) -> List[Dict]:
    """Parse a single Reuters SGML file and return a list of documents."""
    with open(file_path, 'r', encoding='latin-1') as f:
        content = f.read()

    soup = BeautifulSoup(content, 'lxml')
    documents = []

    for article in soup.find_all('reuters'):
        doc_id = article.get('newid', '').strip()
        if not doc_id:
            continue

        title_tag = article.find('title')
        body_tag = article.find('body')

        title = title_tag.get_text(separator=' ') if title_tag else ''
        body = body_tag.get_text(separator=' ') if body_tag else ''

        combined = f"{title} {body}".strip()
        if not combined:
            continue

        documents.append({
            'doc_id': doc_id,
            'text': clean_text(combined)
        })

    return documents


def clean_text(text: str) -> str:
    """Minimal text cleaning: lowercase and collapse whitespace."""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_and_process_reuters(data_path: str) -> List[Dict]:
    """
    Parse all .sgm files in the given directory.

    Args:
        data_path: Path to directory containing Reuters .sgm files.

    Returns:
        List of dicts with keys: 'doc_id', 'text'
    """
    all_documents = []

    sgm_files = sorted([
        f for f in os.listdir(data_path) if f.endswith('.sgm')
    ])

    if not sgm_files:
        print(f"[WARNING] No .sgm files found in: {data_path}")
        return all_documents

    for file_name in sgm_files:
        file_path = os.path.join(data_path, file_name)
        docs = parse_sgm_file(file_path)
        all_documents.extend(docs)
        print(f"[INFO] Parsed {file_name}: {len(docs)} documents")

    print(f"\n[DONE] Total documents loaded: {len(all_documents)}")
    return all_documents


def save_documents(documents: List[Dict], output_path: str) -> None:
    """
    Save processed documents to a JSON file.

    Args:
        documents: List of document dicts.
        output_path: Full path to output .json file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)

    print(f"[SAVED] {len(documents)} documents → {output_path}")


if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print("BASE_DIR", BASE_DIR)
    DATA_RAW = os.path.join(BASE_DIR, 'data', 'raw')
    OUTPUT_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'documents.json')
    print("DATA_RAW", DATA_RAW)
    documents = load_and_process_reuters(DATA_RAW)
    save_documents(documents, OUTPUT_PATH)