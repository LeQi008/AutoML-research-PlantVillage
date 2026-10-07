import json


class BaseAgent:
    def __init__(self, client, instructions, tools, tool_executor, name):
        self.client = client
        self.instructions = instructions
        self.tools = tools
        self.tool_executor = tool_executor
        self.name = name

    def run(self, conversation):
        while True:
            response = self.client.responses.create( # Repeat until no more tool call
                model="gpt-5.6-luna",
                instructions=self.instructions,
                input=conversation,
                tools=self.tools,
            )

            conversation.extend(response.output)

            tool_calls = [
                item for item in response.output
                if item.type == "function_call"
            ]

            if not tool_calls: # if there is no tool call, we're done.
                print(f"\n{self.name}: {response.output_text}\n") # Print statement here so all agents have it
                return response.output_text

            for tool_call in tool_calls:
                print(f"\n[{self.name}] Tool requested: {tool_call.name}")
                print(f"Arguments: {tool_call.arguments}")

                arguments = json.loads(tool_call.arguments)

                result = self.tool_executor(
                    tool_call.name,
                    arguments,
                )

                conversation.append({
                    "type": "function_call_output",
                    "call_id": tool_call.call_id,
                    "output": result,
                })
