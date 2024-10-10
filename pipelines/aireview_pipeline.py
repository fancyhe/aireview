"""
title: aireview Pipeline
author: fancyhe
date: 2024-08-12
version: 0.1
license: 
description: A pipeline for assisting review of complex docs per rubrics by GenAI agents.
environment_variables: OLLAMA_BASE_URL OLLAMA_MODEL OLLAMA_EMBEDDING_MODEL VECTOR_STORE_URL
"""

import asyncio
import os
from typing import Generator, Iterator, List, Union

import requests
from pydantic import BaseModel

from pipelines.aireview.agents import ReviewAgentsGraph, iter_over_async


class Pipeline:
    class Valves(BaseModel):
        OLLAMA_BASE_URL: str = ""
        OLLAMA_MODEL: str = ""
        OLLAMA_EMBEDDING_MODEL: str = ""
        VECTOR_STORE_URL: str = ""
        pass

    def __init__(self):
        # Optionally, you can set the id and name of the pipeline.
        # Best practice is to not specify the id so that it can be automatically inferred from the filename, so that users can install multiple versions of the same pipeline.
        # The identifier must be unique across all pipelines.
        # The identifier must be an alphanumeric string that can include underscores or hyphens. It cannot contain spaces, special characters, slashes, or backslashes.
        self.name = "aireview Pipeline"
        # Reachable localhost
        actual_host = "host.docker.internal" if is_docker() else "localhost"
        self.valves = self.Valves(
            **{
                "OLLAMA_BASE_URL": os.getenv(
                    "OLLAMA_BASE_URL", f"http://{actual_host}:11434"
                ),
                "OLLAMA_MODEL": os.getenv("OLLAMA_MODEL", "llama3.1"),
                "OLLAMA_CONTEXT_LENGTH": os.getenv("OLLAMA_CONTEXT_LENGTH", "16384"),
                "OLLAMA_EMBEDDING_MODEL": os.getenv(
                    "OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"
                ),
                "VECTOR_STORE_URL": os.getenv(
                    "VECTOR_STORE_URL", f"http://{actual_host}:6333"
                ),
            }
        )

        self.graph = ReviewAgentsGraph()
        pass

    async def on_startup(self):
        # This function is called when the server is started.
        print(f"on_startup:{__name__}")
        # Env vars as valves values are overridden by config
        os.environ |= {
            k: v
            for k, v in self.valves.__dict__.items()
            if not (k.startswith("__") and k.endswith("__"))
        }
        # Load data
        self.graph.load()
        self.graph.build()
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        print(f"on_shutdown:{__name__}")
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom pipelines like RAG.
        print(f"pipe:{__name__}")

        OLLAMA_BASE_URL = self.valves.OLLAMA_BASE_URL
        OLLAMA_MODEL = self.valves.OLLAMA_MODEL

        if "user" in body:
            print("######################################")
            print(f'# User: {body["user"]["name"]} ({body["user"]["id"]})')
            print(f"# Message: {user_message}")
            print("######################################")

        # Title generation request, simply use the query from 1st message
        if "RESPOND ONLY WITH THE TITLE TEXT" in user_message:
            try:
                r = requests.post(
                    url=f"{OLLAMA_BASE_URL}/v1/chat/completions",
                    json={**body, "model": OLLAMA_MODEL},
                    stream=False,
                )
                r.raise_for_status()
                if body["stream"]:
                    return r.iter_lines()
                else:
                    return (
                        r.json()["choices"][0]["message"]["content"]
                        if "choices" in r.json()
                        else "Unknown"
                    )
            except Exception as e:
                return f"Error: {e}"

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError as e:
            if str(e).startswith("There is no current event loop in thread"):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            else:
                raise

        return iter_over_async(self.graph.run(user_message), loop)


def is_docker():
    def text_in_file(text, filename):
        try:
            with open(filename, encoding="utf-8") as lines:
                return any(text in line for line in lines)
        except OSError:
            return False

    cgroup = "/proc/self/cgroup"
    return os.path.exists("/.dockerenv") or text_in_file("docker", cgroup)
