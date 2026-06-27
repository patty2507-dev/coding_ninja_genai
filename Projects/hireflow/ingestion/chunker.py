from typing import List

from llama_index.core.schema import Document, TextNode
from llama_index.core.node_parser import SentenceSplitter


def chunk_documents(documents: List[Document]) -> List[TextNode]:
    """
    Split resume documents into overlapping chunks.
    chunk_size=512  — captures full sentences per section
    chunk_overlap=64 — small overlap to avoid cutting mid-sentence
    Returns list of TextNode objects with metadata preserved.
    """
    splitter = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=64,
        paragraph_separator="\n\n",
    )

    nodes = splitter.get_nodes_from_documents(documents)

    # Attach chunk index per source file for context enrichment later
    file_chunk_counter: dict = {}

    for node in nodes:
        source = node.metadata.get("file_name", "unknown")
        file_chunk_counter[source] = file_chunk_counter.get(source, 0) + 1
        node.metadata["chunk_index"] = file_chunk_counter[source]

    print(f"Created {len(nodes)} chunk(s) from {len(documents)} document(s)")
    return nodes