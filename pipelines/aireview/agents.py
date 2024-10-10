"""Review agents
"""

import argparse
import asyncio
import logging
import operator
import os
import uuid
import warnings
from collections import deque
from pathlib import Path
from typing import (
    Annotated,
    AsyncGenerator,
    Generator,
    Iterator,
    List,
    Sequence,
    Tuple,
    TypedDict,
    Union,
)

import yaml
from dotenv import load_dotenv
from langchain.globals import set_debug
from langchain_core._api import LangChainBetaWarning
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.base import BaseMessage
from langchain_core.messages.tool import ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
from langchain_core.tools import Tool, tool
from langchain_ollama import ChatOllama
from langfuse.callback import CallbackHandler
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import InjectedState, ToolNode
from rich.logging import RichHandler

try:  # Run by pipelines
    from pipelines.aireview.retrieval import DocMetaName, DocRetriever, DocSectionType
except ModuleNotFoundError:  # Run directly
    from retrieval import DocMetaName, DocRetriever, DocSectionType

# Ignore LangChain deprecation warnings
warnings.simplefilter("ignore", category=LangChainBetaWarning)

logger = logging.getLogger()
root_level = logging.INFO
logging.getLogger().handlers.clear()
logging.basicConfig(
    level=root_level, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)

# Env vars
load_dotenv(verbose=True)
global_config = {}
if os.getenv("LANGFUSE_PUBLIC_KEY"):
    # Initialize Langfuse handler
    langfuse_handler = CallbackHandler()
    # Global config
    global_config = {"callbacks": [langfuse_handler]}


# Tools
def docstring_parameter(*sub):
    """Able to use parameter in docstring"""

    def dec(obj):
        obj.__doc__ = obj.__doc__.format(*sub)
        return obj

    return dec


# Agent State
class ReviewState(TypedDict):
    """LangGraph state to track document reviews by sections"""

    sections_dict: dict[str, List[str]]
    """Sections names dict { doc_file: [sections] }, set by section list retrieval agent"""

    sections: Annotated[List[str], operator.add]
    """Sections content, append by sections retrieval agent"""

    rubrics: Annotated[List[str], operator.add]
    """Corresponding review rubrics content, append by rubrics retrieval agent"""

    current_file: int
    """Index of file under review, set by loop agent"""

    current_section_index: int
    """Index of section under review, set by loop agent"""

    reviews: Annotated[List[str], operator.add]
    """Review results, appended by review agents"""

    messages: Annotated[Sequence[BaseMessage], operator.add]
    """Converstation messages, appended by (Human, AI, Tool)"""


def retrieve_section_list(state: ReviewState, config) -> ReviewState:
    """Retrieve sections name list for given user message"""

    model = config["configurable"]["model"]
    retriever: DocRetriever = config["configurable"]["retriever"]

    # Retrieval - get section nodes by user message
    user_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    query = user_messages[-1].content

    # 1. Use `header`` instead of `content` to find most relavent doc
    all_sections_dict = retriever.retrieve_section_names_dict(
        query=query, section_type=DocSectionType.HEADER
    )
    logger.info(
        "[retrieve_section_list] Relevant file(s): [%s]", list(all_sections_dict)
    )
    # 2. Find the rubric doc, get all sections
    rubric_docs = retriever.retrieve(query, DocSectionType.RUBRIC, k=1)
    rubric_source = rubric_docs[0].metadata[DocMetaName.SOURCE]
    rubric_sections = retriever.retrieve_doc_section_list(query, rubric_source)
    logger.info(
        "[retrieve_section_list] Relevant rubric(s) [%s] sections: [%s] ... (truncated)",
        rubric_source,
        rubric_sections[:3],
    )
    # 3. Identify the common sections
    # TODO: Handle case of inconsistent sections across files
    all_section_names = sorted(next(iter(all_sections_dict.values())))
    common_section_names = [s for s in all_section_names if s in rubric_sections]
    # Use JSON tool spec instead of @tool annotation, based on test results
    fetch_context_section_names = {
        "name": "fetch_context_section_names",
        "description": "Select the document section names which are necessary to fulfill the review request."
        " If the review request contains section numbers, select only the matching numbers."
        " If the review request describes a topic, select section names that are related to the topic.",
        "parameters": {
            "type": "object",
            "properties": {
                "sections": {
                    "type": "array",
                    "description": "List of document section names",
                    "enum": common_section_names,
                    "items": {"type": "string"},
                }
            },
            "required": ["sections"],
        },
    }
    system_prompt = ""  # Not needed here
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "{system_prompt}"),
            ("placeholder", "{messages}"),
        ]
    )
    messages = state["messages"]

    tools = [fetch_context_section_names]
    model_with_tools = model.bind_tools(tools)
    chain = prompt | model_with_tools
    response: AIMessage = chain.invoke(
        {"system_prompt": system_prompt, "messages": messages}, config=global_config
    )
    section_names = response.tool_calls[0]["args"]["sections"]
    # Workaround fix on list format
    if isinstance(section_names, str):
        section_names = section_names.strip('[]"').split('", "')
        response.tool_calls[0]["args"]["sections"] = section_names
    logger.info("[retrieve_section_list] Section names: %s", section_names)

    sections_dict = {}
    for file in all_sections_dict.keys():
        sections_dict[file] = section_names
    logger.info("[retrieve_section_list] Sections dict: %s", sections_dict)

    return {"sections_dict": sections_dict, "messages": [response]}


def loop_through_sections(state: ReviewState) -> ReviewState:
    """Determine next section to review"""
    dict = state["sections_dict"]
    current_file = state["current_file"]
    if not state["reviews"]:
        next_file = next(iter(dict.keys()))
        next_index = 0
        logger.info(
            "[loop_through_sections] Just get started, file [%s] section [%s]",
            next_file,
            next_index,
        )
    else:
        if state["current_section_index"] + 1 < len(dict[current_file]):
            next_file = current_file
            next_index = state["current_section_index"] + 1
        else:
            # When next is 'None' will be handled in `determine_next_step`
            next_file = next(
                iter(list(dict)[list(dict).index(current_file) + 1 :]), None
            )
            next_index = 0

        logger.info(
            "[loop_through_sections] Proceed to file [%s] section [%s]",
            next_file,
            next_index,
        )
    return {
        "current_file": next_file,
        "current_section_index": next_index,
    }


def determine_next_step(state: ReviewState) -> str:
    """Determine the next step in the workflow"""
    current_file = state["current_file"]

    next_step = "continue" if current_file else "end"

    logger.info(
        "[determine_next_step] Current file: %s. Next step: %s", current_file, next_step
    )
    return next_step


def retrieve_sections(state: ReviewState, config) -> ReviewState:
    """Retrieve sections content for given user message"""

    retriever: DocRetriever = config["configurable"]["retriever"]
    dict = state["sections_dict"]
    doc_file = state["current_file"]
    index = state["current_section_index"]
    section_name = dict[doc_file][index]

    # Retriever
    section_doc = retriever.retrieve_doc_section_direct(
        doc_file, section_name, DocSectionType.CONTENT
    )
    logger.info("[retrieve_sections] docs: %s ... (truncated)", section_doc[:80])

    # Emit a message to show the rubrics to user
    message = AIMessage(content=f"Section for review: \n{section_doc}")

    return {"sections": [section_doc], "messages": [message]}


def retrieve_review_rubrics(state: ReviewState, config) -> ReviewState:
    """Retrieve review rubrics for given user message"""
    retriever: DocRetriever = config["configurable"]["retriever"]
    dict = state["sections_dict"]
    doc_file = state["current_file"]
    index = state["current_section_index"]
    section_name = dict[doc_file][index]

    # Retrieval #1: Get section nodes by user message
    user_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    user_message = user_messages[-1].content
    section_rubrics = retriever.retrieve_doc_section_by_query(
        user_message, section_name, DocSectionType.RUBRIC
    )
    # nodes = retriever.retrieve(query=user_message, section_type=DocSectionType.RUBRIC)
    # Retrieval #2: Get secton content by section names
    logger.info("[retrieve_review_rubrics] Review rubrics: %s", section_rubrics)

    # Emit a message to show the rubrics to user
    message = AIMessage(content=f"Rubrics for sections review: \n{section_rubrics}")
    return {"rubrics": [section_rubrics], "messages": [message]}


@tool(parse_docstring=True, response_format="content_and_artifact")
def tool_explain_terminology(terminology_list: List[str]) -> Tuple[str, List[str]]:
    """Give explanation for a terminology or acronym, abbreviation..

    Args:
        terminology_list: List of Terminology
    """
    # TODO: Return actual explanation via RAG.
    context = f"{terminology_list} means: {terminology_list}"
    logger.info(
        "[explain_terminology] Tool called. Got query param: %s", terminology_list
    )

    message = f"Fetched context for query: {terminology_list}"
    return message, context


@tool(parse_docstring=True, response_format="content_and_artifact")
def tool_fetch_section(
    section_refs: Union[str, List[str]],
    state: Annotated[dict, InjectedState],
    config: RunnableConfig,
) -> Tuple[str, List[str]]:
    """Fetch content of another section.

    Args:
        section_refs: List of section refs. Each section ref is the Id of section (for example, "1.2") or name of section (for example, "Technical Design")
    """
    logger.info("[fetch_other_section] Tool called. Got param: [%s]", section_refs)
    # Workaround fix on list format
    if isinstance(section_refs, str):
        section_refs = section_refs.strip('[]"').split('", "')

    sections_dict: dict[str, List[str]] = state["sections_dict"]
    retriever: DocRetriever = config["configurable"]["retriever"]

    user_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    query = user_messages[-1].content
    all_sections_dict = retriever.retrieve_section_names_dict(
        query=query, section_type=DocSectionType.HEADER
    )

    all_section_names = sorted(next(iter(all_sections_dict.values())))
    section_contents = []
    for section_ref in section_refs:
        section_name = ""
        for s in all_section_names:
            if section_ref in s:
                section_name = s
                break
        section_content = retriever.retrieve_doc_section_direct(
            next(iter(sections_dict.keys())), section_name, DocSectionType.CONTENT
        )
        section_contents.append(section_content)

    message = f"Fetched context for section: {section_refs}"
    logger.info(
        "[tool_fetch_section] content: %s, artifact: %s", message, section_contents
    )
    return message, section_contents


@tool(parse_docstring=True, response_format="content_and_artifact")
def tool_fetch_url(url: List[str]) -> Tuple[str, List[str]]:
    """Fetch external web resource from a specific URL.

    Args:
        url: The URL to fetch
    """
    # TODO: Return actual explanation via retrieval.
    context = f"{url} means: {url}"  # Echo as dummy fetched reference
    logger.info("[fetch_external_url] Tool called. Got query param: %s", url)
    message = f"Fetched context for query: {url}"
    return message, context


async def retrieve_review_context(state: ReviewState, config: RunnableConfig):
    """Agent: "retrieve_review_context" """
    model: ChatOllama = config["configurable"]["model"]

    index = state["current_section_index"]
    docs = state["sections"][index]
    rubrics = state["rubrics"][index]

    system_prompt = ""
    user_prompt = (
        "Determine if the following document can be reviewed with sufficient context. If not, indicate what context is missing, specifically: "
        " 1. Terminology and acronyms."
        " 2. Content of another section."
        " 3. External links."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "{system_prompt}"),
            ("user", "{user_prompt}"),
            ("placeholder", "{messages}"),
            ("user", "[Document] {docs}"),
        ]
    )
    # Tool messages
    messages = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    model_with_tools = model.bind_tools(
        [tool_explain_terminology, tool_fetch_section, tool_fetch_url]
    )
    chain = prompt | model_with_tools
    response: AIMessage = await chain.ainvoke(
        {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "docs": docs,
            "rubrics": rubrics,
            "messages": messages,
        },
        config=global_config,
    )

    # Deduplication for entries in `tool_calls`
    if hasattr(response, "tool_calls"):
        deduplicated = list(
            {(d["name"], tuple(d["args"])): d for d in response.tool_calls}.values()
        )
        response.tool_calls = deduplicated

    return {"messages": [response]}


async def review_section(state: ReviewState, config: RunnableConfig):
    """Agent: "reviewer_agent" """
    model: ChatOllama = config["configurable"]["model"]
    review_system_prompt = config["configurable"]["review_system_prompt"]

    index = state["current_section_index"]
    docs = state["sections"][index]
    rubrics = state["rubrics"][index]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "{system_prompt}"),
            ("placeholder", "{messages}"),
            ("user", "[Document] {docs}"),
            ("user", "[Rubrics] {rubrics}"),
            ("user", "[References] {tool_messages}"),
        ]
    )
    # Extract review requests. Human messages only.
    messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    # Tool call results
    tool_messages = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    ref_list = []
    for m in tool_messages:
        match m.artifact:
            case list():
                ref_list.append(m.content + "\n".join(m.artifact))
            case str():
                ref_list.append(m.content + m.artifact)
            case _:
                ref_list.append(m.content + str(m.artifact))
    reference_context = "\n".join(ref_list)

    chain = prompt | model  # model_with_tools can't stream
    response = await chain.ainvoke(
        {
            "system_prompt": review_system_prompt,
            "messages": messages,
            "docs": docs,
            "rubrics": rubrics,
            "tool_messages": reference_context,
        },
        config=global_config,
    )

    return {"reviews": [response.content], "messages": [response]}


def determine_review_context(state: ReviewState) -> str:
    """Determine if route to tool call needed for review context"""
    # If last message is tool call then call
    messages = state["messages"]
    last_msg = messages[-1] if messages else None
    current_tool_call = {
        call["name"]: call["args"]
        for call in last_msg.tool_calls
        if hasattr(last_msg, "tool_calls")
    }
    # List of ToolMessages
    history_tool_calls = {
        call["name"]: call["args"]
        for m in state["messages"][: len(state["messages"]) - 1]
        if hasattr(m, "tool_calls")
        for call in m.tool_calls
    }

    logger.info(
        "[determine_review_context] Current tool calls: %s",
        current_tool_call,
    )
    logger.info("[determine_review_context] History tool calls: %s", history_tool_calls)

    # If there is new function call, then call tools. Otherwise continue.
    # for call in history_tool_calls:
    next = "done"
    for call_name, call_args in current_tool_call.items():
        if call_name not in history_tool_calls:
            next = "tool_calls"
            break
        if history_tool_calls[call_name] != call_args:
            next = "tool_calls"
            break

    logger.info(
        "[determine_review_context] Based on 'tool_calls' existence in history messages, next: [%s]",
        next,
    )
    return next


def aggregate_review_results(state: ReviewState) -> ReviewState:
    """Aggregrate all review results"""
    # For now just merge section reviews
    agg_reviews = [", ".join(state["reviews"])]
    logger.info(
        "[aggregate_review_results] Aggregrated review results: %s", agg_reviews
    )
    # TODO: Use LLM to summarize

    return {"messages": [AIMessage("Aggregrated reviews")]}


class ReviewAgentsGraph:
    """Review Agents with LangGrpah"""

    def __init__(self):
        # Retriever
        self.r = DocRetriever()
        self.graph = None
        self.model = None
        self.review_system_prompt = None

    def get_model(self):
        if self.model:
            return self.model
        self.model = ChatOllama(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            model=os.getenv("OLLAMA_MODEL", "llama3.1"),
            temperature=0,
            num_ctx=int(os.getenv("OLLAMA_CONTEXT_LENGTH", "16384")),  # default is 8192
            keep_alive="30m",  # keep loaded 30m
        )
        return self.model

    # Load data
    def load(self):
        data_dir = os.environ.get("PIPELINES_DATA_DIR", "data")
        # Load documents
        self.r.load(input_dir=data_dir)
        # Load config
        with open(
            Path.resolve(Path.cwd() / data_dir / "config.yaml"), "r", encoding="utf-8"
        ) as stream:
            config_dict = yaml.safe_load(stream)
        # Review - system prompt
        self.review_system_prompt = config_dict["review"]["system_prompt"]
        logger.info("[Data] Review system prompt: %s", self.review_system_prompt)

    # Build graph
    def build(self):

        # New session ID
        if os.getenv("LANGFUSE_PUBLIC_KEY") and langfuse_handler:
            langfuse_handler.session_id = str(uuid.uuid4())

        tools = [tool_explain_terminology, tool_fetch_section, tool_fetch_url]
        tool_node = ToolNode(tools)
        # Create the graph
        workflow = StateGraph(ReviewState)

        # Add nodes
        workflow.add_node("retrieve_section_list", retrieve_section_list)
        workflow.add_node("retrieve_sections", retrieve_sections)
        workflow.add_node("retrieve_review_rubrics", retrieve_review_rubrics)
        workflow.add_node("retrieve_review_context", retrieve_review_context)
        workflow.add_node("review_section", review_section)
        workflow.add_node("loop_through_sections", loop_through_sections)
        workflow.add_node("aggregate_results", aggregate_review_results)
        workflow.add_node("tool_calls", tool_node)

        # Set up the flow
        workflow.set_entry_point("retrieve_section_list")
        workflow.add_edge("retrieve_section_list", "loop_through_sections")

        workflow.add_conditional_edges(
            "loop_through_sections",
            determine_next_step,
            {
                "continue": "retrieve_sections",
                "end": "aggregate_results",
            },
        )

        workflow.add_edge("retrieve_sections", "retrieve_review_rubrics")
        workflow.add_edge("retrieve_review_rubrics", "retrieve_review_context")

        workflow.add_conditional_edges(
            "retrieve_review_context",
            determine_review_context,
            {
                "tool_calls": "tool_calls",
                "done": "review_section",
            },
        )
        workflow.add_edge("tool_calls", "review_section")
        workflow.add_edge("review_section", "loop_through_sections")
        workflow.add_edge("aggregate_results", END)

        # Compile the graph
        self.graph = workflow.compile()

        self.save_graph_image(self.graph)

    # Run
    async def run(self, user_message: str) -> AsyncGenerator:
        logger.info("User message is: %s", user_message)
        messages = [HumanMessage(user_message)]
        # Initial state
        state = {
            "sections": [],
            "rubrics": [],
            "current_section_index": 0,
            "reviews": [],
            "messages": messages,
        }
        config = global_config | {
            "configurable": {
                "model": self.get_model(),
                "retriever": self.r,
                "review_system_prompt": self.review_system_prompt,
            }
        }
        async for event in self.graph.astream_events(
            state, config=config, version="v2"
        ):
            match event["event"]:
                case "on_chat_model_stream":
                    content = event["data"]["chunk"].content
                    if content:  # If chat model streaming just yield content str
                        yield f"{content}"
                case "on_chain_stream":
                    content = event["data"]["chunk"]
                    if content:
                        # yield f"[on_chain_stream] {content}"
                        # yield str(content)
                        pass
                case "on_tool_start":
                    content = event["data"]
                    name = event.get("name", "")
                    input: dict = content.get("input", "")
                    input_filtered = {k: input[k] for k in input if k != "state"}
                    yield f"\n[on_tool_start]\nname: {name} input: {input_filtered}\n\n---\n\n"
                case "on_tool_end":
                    output: ToolMessage = event["data"]["output"]
                    name = event.get("name", "")
                    content = output.content
                    art = output.artifact
                    artifact = "\n".join(art) if isinstance(art, list) else art
                    yield f"\n[on_tool_end]\nname: {name} content: {content} artifact: \n{artifact}\n\n---\n\n"
                case "on_chain_end":
                    content = event["data"]["output"]
                    if not content or not isinstance(content, dict):
                        continue
                    # Ignore new messages, and reviews (already streamed)
                    content |= {"messages": [], "reviews": []}
                    if state == content:
                        continue
                    for k, v2 in content.items():
                        v1 = state.get(k, "")
                        if v2 != v1:
                            output = v2
                            if not output:
                                continue
                            if isinstance(v2, list):
                                output = "\n" + "\n".join(v2) + "\n"
                            yield f"\n[{k}]: {output}\n\n---\n\n"
                    state |= content
        yield "\n\n---\n"

    def save_graph_image(self, graph):
        """Save graph image to png file. Requires Pyppeteer installation."""
        image_path = "graph.png"
        if Path(image_path).exists():
            logger.info("Skipped graph image creation as [%s] exists.", image_path)
            return
        try:
            graph.get_graph(xray=True).draw_mermaid_png(
                curve_style=CurveStyle.BASIS,
                node_colors=NodeStyles(
                    first="fill:#ffdfba",
                    last="fill:#baffc9",
                    default="fill:#fad7de,line-height:1.2",
                ),
                wrap_label_n_words=9,
                output_file_path=image_path,
                draw_method=MermaidDrawMethod.PYPPETEER,
                background_color="white",
                padding=10,
            )
            logger.info("Graph image created in [%s].", image_path)
        except Exception as e:
            logger.error("Graph image save error %s", e)
            pass


def iter_over_async(ait, loop):
    ait = ait.__aiter__()

    async def get_next():
        try:
            obj = await ait.__anext__()
            return False, obj
        except StopAsyncIteration:
            return True, None

    while True:
        done, obj = loop.run_until_complete(get_next())
        if done:
            break
        yield obj


def main():
    load_dotenv(verbose=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("user_message", help="User query message")
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

    user_message = args.user_message

    graph = ReviewAgentsGraph()
    graph.load()
    graph.build()

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError as e:
        if str(e).startswith("There is no current event loop in thread"):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        else:
            raise

    for item in iter_over_async(graph.run(user_message), loop):
        print(item, end="", flush=True)


if __name__ == "__main__":
    main()
