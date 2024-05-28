from crewai import Agent
from tools import yt_search_tool

from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain_community.chat_models.huggingface import ChatHuggingFace

model = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text_generation",
    max_new_tokens="8192",
    temperature=0.1,
    top_k=5,
    huggingfacehub_api_token="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",
    repetition_penalty=1.03,
)

llm = ChatHuggingFace(llm=model)

# Create a blog content researcher
blog_researcher = Agent(
    role="Senior Blog Researcher from Youtube Videos",
    goal="Get the relevant video content for the topic {topic} from YT channel",
    verbose=True,
    memory=True,
    backstory=(
        "Expert in understandig videos in AI Data Science, Machine Learning and GEN AI and providing suggestion"
    ),
    tools=[yt_search_tool],
    allow_delegation=True,
    llm=llm,
)

blog_writer = Agent(
    role="Blog Writer",
    goal="Narrate compelling tech stories about the video {topic} from YT channel",
    verbose=True,
    memory=True,
    backstory=(
        "With a flair for simplifying complex topics, you craft"
        "engaging narratives that captivate and educate, bringing new"
        "discoveries to light in an accessible manner."
    ),
    tools=[],
    allow_delegation=False,
    llm=llm,
)
