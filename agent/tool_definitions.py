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

DELEGATE_TO_LITERATURE_AGENT_TOOL = {
    "type": "function",
    "name": "delegate_to_literature_agent",
    "description": "Delegate a literature research task to the Literature Agent. The Literature Agent can search the web and save relevant literature findings.",
    "parameters": {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "Detailed literature research question or task.",
            }
        },
        "required": ["task"],
    },
}

#==== Literature tools ====

WEB_SEARCH_TOOL = {
    "type": "web_search",
}

SAVE_LITERATURE_TOOL = {
    "type": "function",
    "name": "save_literature",
    "description": (
        "Save a verified literature finding as a persistent literature record. "
        "The system automatically generates a unique literature ID. Do not "
        "specify or reuse literature IDs."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Title of the paper, article, or source.",
            },
            "authors": {
                "type": "string",
                "description": "Authors or organization responsible for the source.",
            },
            "year": {
                "type": "string",
                "description": "Publication year or date if available.",
            },
            "venue": {
                "type": "string",
                "description": "Journal, conference, repository, or website.",
            },
            "source_url": {
                "type": "string",
                "description": "URL of the source.",
            },
            "summary": {
                "type": "string",
                "description": "Concise summary of the relevant method and findings.",
            },
            "relevance": {
                "type": "string",
                "description": "Why this source is relevant to the current research.",
            },
            "limitations": {
                "type": "string",
                "description": "Important limitations, caveats, or transferability concerns.",
            },
        },
        "required": ["title", "source_url", "summary", "relevance"],
    },
}

READ_LITERATURE_TOOL = {
    "type": "function",
    "name": "read_literature",
    "description": "Retrieve previously saved literature records from persistent literature memory.",
    "parameters": {
        "type": "object",
        "properties": {
            "literature_id": {
                "type": "string",
                "description": "Specific literature ID such as LIT-001.",
            },
            "query": {
                "type": "string",
                "description": "Keyword or phrase to search across saved literature records.",
            },
        },
        "required": [],
    },
}