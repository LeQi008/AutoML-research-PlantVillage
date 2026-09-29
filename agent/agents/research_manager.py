from agents.base_agent import BaseAgent


RESEARCH_MANAGER_INSTRUCTIONS = """
You are the Research Manager of an AutoML research lab.

You are responsible for directing the research process.

Your responsibilities are:
- Understand the user's research objective.
- Inspect the existing source code and experiment history.
- Formulate scientifically meaningful hypotheses.
- Decide what should be investigated next.
- Delegate implementation tasks to the Model Engineer.
- Delegate experiment execution to the Experimenter.
- Analyze returned experimental results.
- Compare results against previous experiments.
- Decide whether to continue, modify, or change the research direction.

You should reason about both model performance and computational cost.

Every experiment must respect the user's stated hardware constraints.

Do not directly modify source files.
Do not directly run experiments.
Delegate those tasks to the appropriate specialist.

When delegating a task, provide enough context for the specialist
to perform the task independently.
"""

MANAGER_TOOLS = [
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
        "name": "list_experiments",
        "description": "List completed AutoML experiments and their summaries.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "type": "function",
        "name": "delegate_to_model_engineer",
        "description": "Delegate a source-code or model implementation task to the Model Engineer.",
        "parameters": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Detailed implementation task.",
                }
            },
            "required": ["task"],
        },
    },
    {
        "type": "function",
        "name": "delegate_to_experimenter",
        "description": "Delegate an experiment execution task to the Experimenter.",
        "parameters": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Detailed experiment request.",
                }
            },
            "required": ["task"],
        },
    },
]


def create_research_manager(client, model_engineer, experimenter, tool_executor):
    def manager_tool_executor(name, arguments):
        if name == "delegate_to_model_engineer":
            return model_engineer.run([
                {
                    "role": "user",
                    "content": arguments["task"],
                }
            ])

        if name == "delegate_to_experimenter":
            return experimenter.run([
                {
                    "role": "user",
                    "content": arguments["task"],
                }
            ])

        # else
        return tool_executor(name, arguments)

    return BaseAgent(
        client=client,
        instructions=RESEARCH_MANAGER_INSTRUCTIONS,
        tools=MANAGER_TOOLS,
        tool_executor=manager_tool_executor,
        name="Research Manager",
    )