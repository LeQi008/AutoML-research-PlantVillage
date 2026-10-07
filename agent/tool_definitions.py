#==== File tools ====

LIST_FILES_TOOL = {
    "type": "function",
    "name": "list_files",
    "description": "List accessible files in the AutoML Research Lab repository.",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

READ_FILE_TOOL = {
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
}

WRITE_FILE_TOOL = {
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
}


#==== Experiment tools ====

LIST_EXPERIMENTS_TOOL = {
    "type": "function",
    "name": "list_experiments",
    "description": "List completed AutoML experiments and their summaries.",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

RUN_EXPERIMENT_TOOL = {
    "type": "function",
    "name": "run_experiment",
    "description": (
        "Run a new experiment using the existing pipeline. "
        "The experiment runner automatically generates a unique experiment ID. "
        "Do not specify or reuse experiment IDs."
    ),
    "parameters": {
        "type": "object",
        "properties": {

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
}


#==== Delegation tools ====

DELEGATE_TO_MODEL_ENGINEER_TOOL = {
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
}

DELEGATE_TO_EXPERIMENTER_TOOL = {
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
}
