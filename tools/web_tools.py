from browser_use import Agent
from langchain_google_genai import ChatGoogleGenerativeAI
import asyncio

async def run_web_research(task_description: str) -> str:
    """
    Uses a headless browser to navigate the web, scrape data, or summarize videos.
    """
    try:
        # Initialize Gemini via Langchain
        # Ensure your GEMINI_API_KEY is in the .env file
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
        
        # Initialize the browser agent
        agent = Agent(
            task=task_description,
            llm=llm
        )
        
        # Run the agent asynchronously
        result = await agent.run()
        
        # browser-use returns a structured object, final_result() extracts the LLM's answer
        return result.final_result()
        
    except Exception as e:
        return f"Web Research Failed: {str(e)}"