from agents.base_agent import BaseAgent


MODEL_ENGINEER_INSTRUCTIONS = """
You are the Model Engineer in an AutoML research lab.

Your responsibility is to implement model and source-code changes
requested by the Research Manager.

You should:
- Inspect the existing source code before making changes.
- Understand the requested hypothesis and implementation requirements.
- Modify the appropriate source files.
- Maintain compatibility with the existing experiment pipeline.
- Report exactly what you changed.

You may use write_file to modify source files.

You do NOT:
- decide the overall research direction
- run experiments
- choose experiments
- modify experiment results
"""

MODEL_ENGINEER_TOOLS = [
    {
        "type": "function",
        "name": "list_files",
        "description": "List accessible files in the AutoML Research Lab repository.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "type": "function",
        "name": "read_file",
        "description": "Read an accessible file from the repository.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path relative to the project root.",
                }
            },
            "required": ["path"],
        },
    },
    {
        "type": "function",
        "name": "write_file",
        "description": "Write or replace an accessible file in the repository.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path relative to the project root.",
                },
                "content": {
                    "type": "string",
                    "description": "Complete new file content.",
                },
            },
            "required": ["path", "content"],
        },
    },
]


def create_model_engineer(client, tool_executor):
    return BaseAgent(
        client=client,
        instructions=MODEL_ENGINEER_INSTRUCTIONS,
        tools=MODEL_ENGINEER_TOOLS,
        tool_executor=tool_executor,
        name="Model Engineer",
    )