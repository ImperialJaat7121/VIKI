import interpreter
import os

interpreter.llm.model = "groq/llama3-70b-8192"

# 2. Safety Settings
# Set to True so it doesn't pause for human confirmation on every step
interpreter.auto_run = True 
# We set safe_mode to ask/off based on trust. 'off' is needed for full autonomy,
# but we enforce safety via the system prompt below.
interpreter.safe_mode = "off" 

def run_local_command(prompt: str) -> str:
    """Executes a command on the local OS using Open Interpreter."""
    try:
        # GET CURRENT DIRECTORY ABSOLUTE PATH
        current_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        workspace_dir = os.path.join(current_dir, "workspace")
        
        # STRICT GUARDRAIL PROMPT
        system_constraint = f"""
        You are a strict OS execution agent. 
        CRITICAL RULES:
        1. You may read files from the Windows user's 'Downloads' folder.
        2. You may ONLY create, edit, or delete files inside this exact path: {workspace_dir}
        3. Do NOT execute any system-level configuration changes.
        """
        
        full_prompt = f"{system_constraint}\n\nTask: {prompt}"
        
        # Execute the prompt
        result = interpreter.chat(full_prompt)
        
        # Open interpreter returns a list of dictionaries (the chat history). 
        # We convert it to a string so LangGraph can store it in the state.
        return str(result)
        
    except Exception as e:
        return f"OS Task Failed: {str(e)}"