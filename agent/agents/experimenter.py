from agents.base_agent import BaseAgent

from tool_definitions import (
    LIST_EXPERIMENTS_TOOL,
    READ_FILE_TOOL,
    RUN_EXPERIMENT_TOOL,
)


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
    LIST_EXPERIMENTS_TOOL,
    READ_FILE_TOOL,
    RUN_EXPERIMENT_TOOL,
]


def create_experimenter(client, tool_executor):
    return BaseAgent(
        client=client,
        instructions=EXPERIMENTER_INSTRUCTIONS,
        tools=EXPERIMENTER_TOOLS,
        tool_executor=tool_executor,
        name="Experimenter",
    )