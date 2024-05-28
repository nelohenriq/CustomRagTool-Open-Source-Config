from crewai_tools import YoutubeChannelSearchTool

from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain_groq import ChatGroq
from decouple import config

from crewai_tools.tools.base_tool import BaseTool
from embedchain import App
from embedchain.config import AppConfig, BaseEmbedderConfig
from embedchain.embedder.huggingface import HuggingFaceEmbedder
from crewai_tools.adapters.embedchain_adapter import EmbedchainAdapter
from typing import Dict, Any

import os

os.environ["HUGGINGFACE_ACCESS_TOKEN"] = config("HUGGINGFACE_ACCESS_TOKEN")

""" model = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text_generation",
    max_new_tokens=8192,
    temperature=0.1,
    top_k=5,
    huggingfacehub_api_token="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",
    repetition_penalty=1.03,
)

llm = ChatHuggingFace(llm=model) """

llm = ChatGroq(
    temperature=0,
    groq_api_key="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",
    model_name="llama3-8b-8192",
    max_tokens=8192,
)


class CustomRagTool(EmbedchainAdapter):
    """A custom RagTool that manages Embedchain configuration internally."""

    def __init__(
        self,
        name: str = "Knowledge base",
        description: str = "A knowledge base that can be used to answer questions.",
        summarize: bool = False,
        chunk_size: int = 500,
        vector_store_type: str = "chromadb",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        **kwargs: Dict[str, Any],  # Explicit type hinting for kwargs
    ):
        # Create HuggingFace embedder config
        hf_config = BaseEmbedderConfig(model=embedding_model)
        embedder = HuggingFaceEmbedder(config=hf_config)

        # Create Embedchain app with explicit configuration
        app_config = AppConfig()
        app = App(config=app_config, embedding_model=embedder)

        # Initialize the EmbedchainAdapter with the Embedchain app
        super().__init__(embedchain_app=app, summarize=summarize)

    def _run(self, query: str, **kwargs: Dict[str, Any]) -> str:
        """
        Concrete implementation of the _run method.
        This will fetch and return relevant content from the knowledge base.
        """
        self._before_run(query, **kwargs)  # Perform any pre-processing if needed

        # Fetch relevant content using the adapter
        relevant_content = self.adapter.query(query)
        return f"Relevant Content:\n{relevant_content}"


# Initialize the tool
yt_search_tool = YoutubeChannelSearchTool(
    config=dict(
        llm=llm,
        embedder=dict(
            provider="huggingface",
            config=dict(
                model="all-MiniLM-L6-v2",
            ),
        ),
    ),
    youtube_channel_handle="@krishnaik06",
    adapter=CustomRagTool(
        name="YouTube Search Tool",
        description="Tool to search and summarize YouTube channel content.",
    ),
)
