# This file is the endpoint for agent orchestration

from dotenv import load_dotenv
from openai import OpenAI

from tools import (
    list_files,
    list_experiments,
    read_file,
    run_experiment,
    write_file,
)

from agents.model_engineer import create_model_engineer
from agents.experimenter import create_experimenter
from agents.research_manager import create_research_manager


load_dotenv()

client = OpenAI()


def execute_tool(name, arguments):
    if name == "list_files":
        return list_files()

    if name == "read_file":
        return read_file(arguments["path"])

    if name == "write_file":
        return write_file(
            arguments["path"],
            arguments["content"],
        )

    if name == "list_experiments":
        return list_experiments()

    if name == "run_experiment":
        return run_experiment(
            objective=arguments.get("objective"),
            epochs=arguments.get("epochs"),
            learning_rate=arguments.get("learning_rate"),
            batch_size=arguments.get("batch_size"),
            model_name=arguments.get("model_name"),
            scheduler=arguments.get("scheduler"),
        )

    return f"Unknown tool: {name}"


def main():
    model_engineer = create_model_engineer(
        client,
        execute_tool,
    )

    experimenter = create_experimenter(
        client,
        execute_tool,
    )

    research_manager = create_research_manager(
        client,
        model_engineer,
        experimenter,
        execute_tool,
    )

    print("AutoML Research Lab")
    print("Research Manager ready.")
    print("Type 'exit' to quit.\n")

    conversation = []

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            break

        conversation.append({
            "role": "user",
            "content": user_input,
        })

        response = research_manager.run(conversation)

        conversation.append({
            "role": "assistant",
            "content": response,
        })

        print(f"\nResearch Manager: {response}\n")


if __name__ == "__main__":
    main()
