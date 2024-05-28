from crewai import Task
from tools import yt_search_tool
from agents import blog_researcher, blog_writer

# Research Task
research_task = Task(
    description=(
        'Identify the video {topic}'
        'Get detailed information about the video from the channel'
    ),
    expected_output='A comprehensive 3 paragraphs long report based on the {topic} of the video content',
    tools=[yt_search_tool],
    agent=blog_researcher
)

# Writing Task
write_task = Task(
    description=(
        'Get the info from the youtube channel on the topic {topic}.'
    ),
    expected_output='Summarize the info from the youtube channel video on topic {topic} and create the content for the blog',
    tools=[yt_search_tool],
    agent=blog_writer,
    async_execution=False,
    output_file='new-blog-post.md'
)