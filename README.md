# aireview

Assist review of complex docs per rubrics by GenAI agents.

## Challenges

LLM can do review for single section of document, when instructions and rubrics are given, and there's no dependency on external contexts.

Challenges for review actual documents (e.g. architecture doc, high/low level design doc):

#### 1. Content Size

A complete documentation can have many sections in a hierarchy, meanwhile the size of document can exceed supported context size of LLM.

#### 2. Mixture of Content Types

Including:

* Plain text
* Table (Markdown, HTML, etc.)
* Structured data (JSON, Yaml)
* Source code (actual or pseudo)
* Diagrams

#### 3. External References

Including:

* URL link to external sites
* Anchor link to other sections in same doc
* Mention of section or title of external doc
* Acronym or abbreviation

## Data Security

All components running at local while no data leaves the local machine.

To further ensure no external network access from the containers, possible security hardening can be done:
* Use only the `internal` Docker network, or disable external access & NAT on default `bridge` network
* Utilize local application firewall software for outbound traffic such as [Little Snitch](https://obdev.at/products/littlesnitch/index.html) or [LuLu](https://objective-see.org/products/lulu.html) for macOS.


## Components

| Component Role | Name | License | Description |
| --- | --- | --- | --- |
| LLM Inference | [Ollama](https://github.com/ollama/ollama) | MIT | To run LLM inference with local GPU |
| LLM | [llama3.1:8b](https://ollama.com/library/llama3.1) | Meta Llama 3 Community | Local LLM with Function Calling capabilities |
| Embedding | [nomic-embed-text](https://ollama.com/library/nomic-embed-text) | Apache 2.0 | A high-performing open embedding model with a large token context window. |
| Agent Framework | [LangGraph](https://github.com/langchain-ai/langgraph) | MIT | Library for building stateful, multi-actor applications with LLMs, to create multi-agent workflows. |
| Web UI | [open-webui](https://github.com/open-webui/open-webui) | MIT | Extensible, feature-rich, and user-friendly self-hosted WebUI designed to operate entirely offline. |
| Web UI Integration | open-webui [pipelines](https://github.com/open-webui/pipelines) | MIT | Versatile, UI-Agnostic OpenAI-Compatible Plugin Framework |


## Agents Workflow

Generated LangGraph graph:

<div align="center">
<image src="./graph.png" width="400"/>
</div>

An agents based workflow is used to address these challenges in review

0. The document under review be placed in data folder.
1. User (the human reviewer) send query to hint what sections to review. Examples can be:

    - "Review X architecture doc section 5.10, 6.1"
    - "Review X design doc sections from 5.11 to 5.15"
    - "Review X design doc sections related to software supply chain security"

2. The query will be send to RAG agent for retrieval of most relevant sections, as well as avilable reference instructions and review rubrics. With extra processing:

    - Ensuring mentioned section IDs 
    - Ensuring correctness of section names

3. Reviewer agent calls the Tool agent to fetch addtional references when needed

4. In a loop, reviewer agent reviews on each section with aggregated avilable info including:

    - Document section itself
    - Fetched content for remote context (e.g. URLs)
    - Document section guidance (embedded or external)
    - Document review rubrics
    - Section related context (references)

5. During review process, the introduced references and review output are streamed in the web UI chat session.

## Usage

### Setup

Following setup steps are for Apple Silicon Macs with at least 16GB RAM. For Linux and Windows PCs the steps should be similar. 

#### 0. Working directory.

First cloned repository and change working directory to the home of cloned repository.

```shell
git clone git@github.com:fancyhe/aireview.git
cd aireview
```

#### 1. Install and run Ollama

Refer to [Ollama](https://github.com/ollama/ollama).

Pull models:

```shell
# llama3.1:8b
ollama pull llama3.1
# Embedding
ollama pull nomic-embed-text
```

#### 2. Install container runtime and Docker compose

Can be either minimal setup like [colima](https://github.com/abiosoft/colima) or [Podman](https://podman-desktop.io/), [Rancher Desktop](https://rancherdesktop.io/).

Install Docker compose (without Docker Desktop), or equivalent such as [podman-compose](https://github.com/containers/podman-compose).

```shell
brew install docker-compose
```

Configure docker to find the plug-in. For more details run `brew info docker-compose`.

```shell
cat <<< $(jq '.cliPluginsExtraDirs=["/opt/homebrew/lib/docker/cli-plugins"]' ~/.docker/config.json) > ~/.docker/config.json
```

#### 3. Bring up containers

Determine the hostname or IP address used to access host from container. For Docker it's usually `host.docker.internal` on macOS and Linux. For Podman it can be `host.containers.internal`. If using non-default value, set an environment variable before running docker compose:

```shell
# If needs to override default value `host.docker.internal`
# export HOST_INTERNAL=...
```

Bring up the containers: `open-webui`, `pipelines` and `qdrant`:


```shell
mkdir -p data
docker compose up -d
```

Upon first start, the pipelines container needs some time to:
1. Install Python dependency packages
2. Populate the Qdrant vector store for documents in `data` folder


To verify the pipelines container successfully loaded the pipelines:

```shell
docker logs -n 20 pipelines
```

When ready, the last log output should be:

```log
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:9099 (Press CTRL+C to quit)
```

For more details on `open-webui` and `pipelines`, refer to `open-webui` [Installation with Default Configuration](https://docs.openwebui.com/getting-started/#installation-with-default-configuration) and `pipelines` [Quick Start with Docker](https://docs.openwebui.com/pipelines/#-quick-start-with-docker).

Qdrant is needed for better retrieval performance needed when large amount of documents in `data` folder. Recommend to use a local Qdrant container. More details in https://qdrant.tech/documentation/quick-start/.


#### 4. Verify open-webui & pipelines configurations

Open brower to visit http://localhost:3000/. Register an account to continue.

In the "Select a model" selection drop-down list, select "aireview Pipeline".

If the pipeline is not available, go to Admin Panel at http://localhost:3000/admin/settings, open `Connections` settings, set them as below:

> `host.docker.internal` is the built-in domain name for Docker container to access host. It might be different for other container engines, e.g. Podman also supports `host.containers.internal`.

| Setting Item | Value                               |
| ------------ | ----------------------------------- |
| OpenAI API   | `http://host.docker.internal:9099`  |
| API Token    | `0p3n-w3bu!`                        |
| Ollama API   | `http://host.docker.internal:11434` |

Open `Pipelines` settings, switch `Pipelines Valves` to the expected pipeline and configure values in valve.

#### 5. Updates

In future when updates available for images or containers config, shutdown, pull latest and bring back on the containers:

```shell
docker compose down
# Fetch the latest changes
git pull
# Pull latest images
docker compose pull

docker compose up -d
```



### Load Data

Copy the following files into `pipelines/data` folder:

| Content             | Required Format |
| ------------------- | --------------- |
| Documents to review | Markdown        |
| Review rubrics      | Markdown        |
| Reference documents | Any             |

Document structure requirements:

* The document sections should be Markdown headers
* The review rubrics sections should match review document's sections, names of headers should contain corresponding headers in review document.

### Start Review

Start a new chat at http://localhost:3000/

From 'Select a model' dropdown list, select the `aireview Pipeline`.

Start review by specifying section numbers or titles. For example:

```markdown
# For single section
Review X architecture doc section 5.10
# For list of sections
Review X architecture doc section 5.10, 6.1
Review X design doc sections from 5.11 to 5.15
# For sections of topics
Review X design doc sections related to software supply chain security
```

## Troubleshooting

### Open-webui pipelines startup

When pipeline failed to load, it'll be moved into a `failed` subfolder. To reload, fix the underlying issue and restart the container.

```shell
mv pipelines/failed/* pipelines/
docker restart pipelines
# Wait till container restarts
docker logs -f -n 20 pipelines
```


## Development

### Open-webui pipelines

Option 1: Run pipelines container manually

After [3. Bring up containers](#3-bring-up-containers) is done, stop the pipelines container and manually run it with overriden entrypoint.

```shell
docker compose down pipelines && docker rm pipelines
docker run -it --rm --name pipelines -p 9099:9099 \
    -v $(pwd)/pipelines:/app/pipelines \
    -v $(pwd)/data:/app/data \
    --entrypoint /bin/bash \
    ghcr.io/open-webui/pipelines:main
```

Within container:

```shell
pip install -r pipelines/aireview/requirements.txt
uvicorn main:app --host "0.0.0.0" --port "9099" --forwarded-allow-ips '*'
```

Option 2: Run pipelines as container. Use pipelines REST API to reload the pipelines:

```shell
curl -X 'POST' \
  'http://localhost:9099/pipelines/reload' \
  -H 'accept: application/json' \
  -H 'Authorization: Bearer 0p3n-w3bu!' \
  -d ''
```

Option 3: Run pipelines as Python process on host. Restart the process to load changes.

```shell
# Virtual env
python3.12 -m venv venv
. venv/bin/activate

export PIPELINES_RUNTIME="$(pwd)/../pipelines-runtime"
mkdir -p $PIPELINES_RUNTIME
git -C $PIPELINES_RUNTIME clone https://github.com/open-webui/pipelines.git
pip install -r ${PIPELINES_RUNTIME}/requirements.txt

```

```shell
export PIPELINES_DIR=$(pwd)/pipelines
sh $PIPELINES_RUNTIME/start.sh
```
