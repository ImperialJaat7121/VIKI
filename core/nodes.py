import asyncio
import base64
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from core.state import CopilotState
from tools.os_tools import run_local_command
from tools.web_tools import run_web_research

async def orchestrator_node(state: CopilotState) -> dict:
    """
    Parses the user prompt and runs the necessary agents in parallel.
    """
    prompt = state["user_prompt"].lower()
    
    async def web_task():
        # Trigger the Gemini Web Researcher if asked to summarize or search
        if "summarize" in prompt or "youtube" in prompt or "search" in prompt:
            print("[System] -> Dispatching Web Agent...")
            return await run_web_research(state["user_prompt"])
        return "No web research requested."
        
    async def os_task():
        # Trigger the local Groq OS Agent if asked to build a UI from a file
        if "website" in prompt or "file" in prompt or "image" in prompt:
            print("[System] -> Dispatching OS Agent...")
            command = "Find the most recently downloaded image in the Windows Downloads folder and return ONLY its absolute path. Do not explain, just return the path."
            # Run the synchronous OS command in a separate thread so it doesn't block the async loop
            return await asyncio.to_thread(run_local_command, command)
        return ""

    # Execute both specialized agents at the exact same time
    web_result, os_result = await asyncio.gather(web_task(), os_task())
    
    # Return the partial state update. LangGraph merges this automatically.
    return {
        "web_research_summary": web_result,
        "local_file_paths": [os_result] if os_result and "No web research" not in os_result else [],
        "current_status": "Information Gathering Complete"
    }

async def developer_node(state: CopilotState) -> dict:
    """
    The Vision-to-Code Agent. Only runs if an image is provided.
    """
    image_paths = state.get("local_file_paths", [])
    
    # Clean the path string just in case the Groq LLM returned extra quotes
    raw_path = image_paths[0].strip("'\" \n") if image_paths else None
        
    if not raw_path or not os.path.exists(raw_path):
        print(f"[System] -> Invalid or missing image path ({raw_path}). Skipping Developer Agent.")
        return {"generated_code": "No code generated.", "current_status": "Skipped Development"}
        
    print(f"[System] -> Passing image ({raw_path}) to GPT-4o Developer Agent...")
    
    # Encode the image for OpenAI
    with open(raw_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
    llm = ChatOpenAI(model="gpt-4o", max_tokens=2500)
    
    message = HumanMessage(
        content=[
            {"type": "text", "text": "You are an expert React and Tailwind developer. Build a single-page UI that perfectly matches this design. Output ONLY the raw HTML/JS code."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]
    )
    
    response = await llm.ainvoke([message])
    generated_code = response.content
    
    print("[System] -> Code generated. Asking OS Agent to save it to the workspace...")
    
    # Use the OS agent to safely save the code to your workspace folder
    current_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    workspace_file = os.path.join(current_dir, "workspace", "index.html")
    
    save_command = f"Create a file at exactly {workspace_file} and write this code into it:\n\n{generated_code}"
    await asyncio.to_thread(run_local_command, save_command)
    
    return {
        "generated_code": f"Successfully written to {workspace_file}",
        "current_status": "Task Complete"
    }