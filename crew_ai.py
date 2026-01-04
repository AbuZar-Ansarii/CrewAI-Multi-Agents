from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv
load_dotenv()
import os
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0
)
print(os.getenv("GEMINI_API_KEY"))

research_agent = Agent(
    role="Research Analyst",
    goal="Collect accurate and relevant information",
    backstory="You are a senior research analyst skilled at extracting insights.",
    llm=llm,
    verbose=True
)

research_task = Task(
    description="Research recent trends in Generative AI",
    expected_output="A concise summary of top GenAI trends",
    agent=research_agent
)

crew = Crew(
    agents=[research_agent],
    tasks=[research_task],
    process=Process.sequential
)

result = crew.kickoff()
print(result)
