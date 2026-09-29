from agents.base_agent import BaseAgent


EXPERIMENTER_INSTRUCTIONS = """
You are the Experimenter in an AutoML research lab.

Your responsibility is to execute experiments requested by the
Research Manager and report their results.

You should:
- Run the requested experiment using the existing experiment pipeline.
- Inspect the resulting experiment files when necessary.
- Report the experiment ID.
- Report configuration and relevant metrics.
- Report Macro F1, accuracy, runtime, and model parameter count
  when available.
- Respect the requested hardware constraints.
- For a rerun, pass the requested config path and experiment ID explicitly
  to run_experiment (for example, config=experiments/EXP-005/config.yaml
  and experiment_id=EXP-005). This is intentional authorization to reuse
  that experiment directory; do not infer a different ID.

You do NOT:
- modify source code
- decide the overall research direction
- independently choose a new architecture
"""

EXPERIMENTER_TOOLS = [
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
        "name": "run_experiment",
        "description": (
            "Run an experiment using the existing pipeline. Leave config and "
            "experiment_id unset for a new legacy-style experiment. For an "
            "explicit rerun, provide both values (for example, "
            "experiments/EXP-005/config.yaml and EXP-005)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "config": {
                    "type": "string",
                    "description": "Config path relative to the project root.",
                },
                "experiment_id": {
                    "type": "string",
                    "description": "Explicit experiment ID, e.g. EXP-005.",
                },
                "objective": {
                    "type": "string",
                    "description": "Objective or hypothesis being tested.",
                },
                "epochs": {
                    "type": "integer",
                    "description": "Number of training epochs.",
                },
                "learning_rate": {
                    "type": "number",
                    "description": "Learning rate.",
                },
                "batch_size": {
                    "type": "integer",
                    "description": "Training batch size.",
                },
                "model_name": {
                    "type": "string",
                    "description": "Model name to use.",
                },
                "scheduler": {
                    "type": "string",
                    "description": "Learning-rate scheduler: none, cosine, or step.",
                },
            },
            "required": [],
        },
    },
]


def create_experimenter(client, tool_executor):
    return BaseAgent(
        client=client,
        instructions=EXPERIMENTER_INSTRUCTIONS,
        tools=EXPERIMENTER_TOOLS,
        tool_executor=tool_executor,
        name="Experimenter",
    )
