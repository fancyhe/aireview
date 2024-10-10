import argparse
import logging
import os
import re
import socket
import sys
import warnings
from dataclasses import dataclass, fields
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from langchain.globals import set_debug, set_verbose
from langchain.indexes import SQLRecordManager, index
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_text_splitters.markdown import ExperimentalMarkdownSyntaxTextSplitter
from qdrant_client import QdrantClient, models, qdrant_client
from rich.logging import RichHandler
from rich.pretty import pretty_repr

logger = logging.getLogger()
root_level = logging.INFO
logging.getLogger().handlers.clear()
logging.basicConfig(
    level=root_level, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)


@dataclass(frozen=True)
class DocMetaName:
    SOURCE: str = "source"  # Source file (relative) path
    SECTION: str = "section"  # Section (Markdown header) name
    TYPE: str = "type"  # Section content type: DocSectionType


@dataclass(frozen=True)
class DocSectionType:
    """Type of section text in review"""

    HEADER: str = "header"  # doc Yaml header
    CONTENT: str = "content"  # submitted content undre view
    INSTRUCTION: str = "instruction"  # instructions for content composing
    RUBRIC: str = "rubric"  # rubrics for content evaluation


class DocMetaMarkdownHeaderTextSplitter(ExperimentalMarkdownSyntaxTextSplitter):
    """Inherient"""

    def get_doc_yaml_header(self, text: str):
        """Get Yaml metadata from Markdown"""

        yaml_header_regex = r"^---\n(.*?)\n---\n"
        match = re.search(yaml_header_regex, text, re.DOTALL)
        if match:
            yaml_header = match.group(1)
            return yaml_header
        else:
            return None

    def split_text(self, text: str, metadata: dict) -> List[Document]:
        """Receives additional metadata and inject in chunks
        'source': 'data/filename.md'
        """

        # Headers
        docs = []
        header_content = self.get_doc_yaml_header(text)
        if header_content:
            yaml_header = Document(
                page_content=header_content,
                metadata=metadata | {DocMetaName.TYPE: DocSectionType.HEADER},
            )

            docs.append(yaml_header)

        # Content sections
        contents = super().split_text(text)
        known_sections = []
        for doc in contents:
            # De-duplication, as heading 1 & 2 'section' overlaps
            if DocMetaName.SECTION in doc.metadata:
                if doc.metadata[DocMetaName.SECTION] in known_sections:
                    continue
                known_sections.append(doc.metadata[DocMetaName.SECTION])
            # Merge doc file metadata (e.g. 'source') in doc section chunk
            doc.metadata |= metadata
            # Add meta about 'section_type'
            if "rubric" in doc.metadata[DocMetaName.SOURCE].lower():
                doc.metadata[DocMetaName.TYPE] = DocSectionType.RUBRIC
            else:
                doc.metadata[DocMetaName.TYPE] = DocSectionType.CONTENT
            docs.append(doc)

        return docs


class DocRetriever:
    def __init__(self):
        self.client = None
        self.embedding = None
        self.vector_store_url = None
        self.record_manager = None
        self.vectorstore: QdrantVectorStore = None
        self.docs: List[Document] = []

    def sync_docs_to_vectorstore(
        self,
        docs: List[Document],
        collection_name: str,
        record_manager: SQLRecordManager,
    ) -> QdrantVectorStore:
        logger.info(
            "Syncing [%s] docs into collection [%s]", len(docs), collection_name
        )
        if not self.client.collection_exists(collection_name):
            logger.info("Vector store collection creating new [%s]", collection_name)
            vectorstore = QdrantVectorStore.from_documents(
                [],
                self.embedding,
                url=self.vector_store_url,
                # prefer_grpc=True,
                collection_name=collection_name,
            )

        logger.info("Vector store collection try loading from [%s]", collection_name)
        vectorstore = QdrantVectorStore(
            client=self.client,
            embedding=self.embedding,
            collection_name=collection_name,
        )

        logger.info("Vector store syncing with cleanup mode 'full'")
        index_status = index(
            docs,
            record_manager,
            vectorstore,
            cleanup="full",
            source_id_key="source",
        )
        logger.info("Vector store sync status: %s", index_status)
        return vectorstore

    def load(self, input_dir: str, force_reload: bool = False):

        self.vector_store_url = os.getenv("VECTOR_STORE_URL", "http://localhost:6333")
        self.client = QdrantClient(location=self.vector_store_url)

        collection_name = "aireview"

        namespace = f"qdrant/{collection_name}"

        self.record_manager = SQLRecordManager(
            f"{namespace}", db_url="sqlite:///record_manager_cache.sql"
        )
        self.record_manager.create_schema()

        self.embedding = OllamaEmbeddings(
            model=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )

        loader = DirectoryLoader(
            input_dir,
            glob="**/*.md",
            use_multithreading=True,
            loader_cls=TextLoader,
            exclude="**/README.md",
        )
        doc_files = loader.load()

        headers_to_split_on = [
            ("#", "section"),
            ("##", "section"),
            ("###", "section"),
        ]

        for df in doc_files:
            markdown_splitter = DocMetaMarkdownHeaderTextSplitter(
                headers_to_split_on=headers_to_split_on, strip_headers=False
            )
            chunks = markdown_splitter.split_text(df.page_content, df.metadata)
            self.docs.extend(chunks)
            logger.info(
                "Added chunks: file [%s] len [%s]",
                df.metadata.get(DocMetaName.SOURCE, ""),
                len(chunks),
            )

        self.vectorstore = self.sync_docs_to_vectorstore(
            self.docs, collection_name, self.record_manager
        )

    def retrieve(
        self,
        query: str,
        section_type: str,
        k: int = 4,
        meta_filter: str = None,
    ) -> list[Document]:
        vectorstore = self.vectorstore
        logger.info(
            "Query: ['%s'] type [%s] meta filter [%s]", query, section_type, meta_filter
        )

        filter = models.Filter(
            must=[
                models.FieldCondition(
                    key=f"metadata.{DocMetaName.TYPE}",
                    match=models.MatchValue(value=section_type),
                )
            ],
        )
        if meta_filter:
            filter_key, filter_value = meta_filter.split("=")
            filter.must.append(
                models.FieldCondition(
                    key=f"metadata.{filter_key}",
                    match=models.MatchValue(value=filter_value),
                )
            )
        docs = vectorstore.similarity_search(query, k=k, filter=filter)
        metadata = [r.metadata for r in docs]
        logger.info(pretty_repr(metadata))

        return docs

    def retrieve_doc_section_list(self, query: str, source: str) -> list[str]:
        """Given doc file source, return list of section names"""
        filter = models.Filter(
            must=[
                models.FieldCondition(
                    key=f"metadata.{DocMetaName.SOURCE}",
                    match=models.MatchValue(value=source),
                )
            ],
        )
        docs = self.vectorstore.similarity_search(query, k=100, filter=filter)
        sections = sorted(
            list(
                dict.fromkeys(
                    [
                        d.metadata[DocMetaName.SECTION]
                        for d in docs
                        if DocMetaName.SECTION in d.metadata
                    ]
                )
            )
        )
        logger.info(
            "Source [%s] has [%s] sections: %s ... (truncated)",
            source,
            len(sections),
            pretty_repr(sections[:3]),
        )
        return sections

    def retrieve_section_names_dict(
        self, query: str, section_type: DocSectionType
    ) -> dict[str, list[str]]:
        """Given query and section_type, return list of section names from `section` metadata"""

        # Retrieve #1 - docs
        docs = self.retrieve(query, section_type)

        # Retrieve #2 - all sections in same 'source' file
        sources = [r.metadata[DocMetaName.SOURCE] for r in docs]
        logger.info("Document sources: %s", pretty_repr(sources))
        # TODO: Hanlde multiple source files
        source = sources[0]  # "data/file.md"
        sections = self.retrieve_doc_section_list(query, source)

        return {source: sections}

    def retrieve_doc_section_direct(
        self, source: str, section: str, section_type: DocSectionType
    ) -> str:
        """Given file name and section name, return section content"""

        filter = models.Filter(
            must=[
                models.FieldCondition(
                    key=f"metadata.{DocMetaName.SOURCE}",
                    match=models.MatchValue(value=source),
                ),
                models.FieldCondition(
                    key=f"metadata.{DocMetaName.SECTION}",
                    match=models.MatchValue(value=section),
                ),
            ],
        )
        docs = self.vectorstore.similarity_search("", k=100, filter=filter)
        # Merge if really multiple
        return "\n".join([doc.page_content for doc in docs])

    def retrieve_doc_section_by_query(
        self, query: str, section: str, section_type: DocSectionType
    ) -> str:
        """Given query, section name and section type, return section content"""

        # Retrieve #1 - docs
        docs = self.retrieve(query, section_type, k=1)
        # Retrieve #2 - sections
        sources = [r.metadata[DocMetaName.SOURCE] for r in docs]
        logger.info("Document sources: %s", pretty_repr(sources))
        # TODO: Hanlde multiple source files
        source = sources[0]  # "data/file.md"

        return self.retrieve_doc_section_direct(source, section, section_type)


def main():
    load_dotenv(verbose=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("user_message", help="User query message")
    parser.add_argument(
        "-t",
        "--type",
        help="Section type: %s" % [f.default for f in fields(DocSectionType)],
    )
    parser.add_argument(
        "-f",
        "--filter",
        help="Filter to apply for `metadata`, in format of 'key=value', like: 'section=name', 'source=file'",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Verbose output. Twice for debug.",
    )
    args = parser.parse_args()

    root_level = logging.INFO
    match args.verbose:
        case v if 1 <= v < 2:
            set_debug(True)
        case v if 2 <= v < 3:
            set_debug(True)
            root_level = logging.DEBUG
    # Reconfigure logging for global
    logging.getLogger().handlers.clear()
    logging.basicConfig(
        level=root_level, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
    )

    query = args.user_message
    type = args.type
    meta_filter = args.filter

    r = DocRetriever()
    r.load(input_dir=os.environ.get("PIPELINES_DATA_DIR", "data"))
    nodes = r.retrieve(query=query, section_type=type, meta_filter=meta_filter)


if __name__ == "__main__":
    main()
