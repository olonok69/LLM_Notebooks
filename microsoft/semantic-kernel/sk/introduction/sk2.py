import asyncio
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import (
    AzureChatCompletion,
)

kernel = Kernel()

chat_completion = AzureChatCompletion(
    service_id="chat-gpt", env_file_path=".env", deployment_name="gpt-4o-chat"
)

# Prepare OpenAI service using credentials stored in the `.env` file
service_id = "chat-gpt"
kernel.add_service(chat_completion)

# Define the request settings
req_settings = kernel.get_prompt_execution_settings_from_service_id(service_id)
req_settings.max_tokens = 2000
req_settings.temperature = 0.4
req_settings.top_p = 0.8

# Create a reusable function summarize function
# TLDR --> Too Long Didn't Read
summarize = kernel.add_function(
    function_name="tldr_function",
    plugin_name="tldr_plugin",
    prompt="{{$input}}\n\nOne line TLDR with the fewest words.",
    prompt_template_settings=req_settings,
)


# Run your prompt
# Note: functions are run asynchronously
async def main():
    # Summarize the laws of thermodynamics
    print(
        await kernel.invoke(
            summarize,
            input="""1st Law of Thermodynamics - Energy cannot be created or destroyed.
                              2nd Law of Thermodynamics - For a spontaneous process, the entropy of the universe increases.
                              3rd Law of Thermodynamics - A perfect crystal at zero Kelvin has zero entropy.""",
        )
    )
    print("\n\n")
    # Summarize the laws of motion
    print(
        await kernel.invoke(
            summarize,
            input="""
    1. An object at rest remains at rest, and an object in motion remains in motion at constant speed and in a straight line unless acted on by an unbalanced force.
    2. The acceleration of an object depends on the mass of the object and the amount of force applied.
    3. Whenever one object exerts a force on another object, the second object exerts an equal and opposite on the first.""",
        )
    )
    print("\n\n")
    # Summarize the law of universal gravitation
    print(
        await kernel.invoke(
            summarize,
            input="""
    Every point mass attracts every single other point mass by a force acting along the line intersecting both points.
    The force is proportional to the product of the two masses and inversely proportional to the square of the distance between them.""",
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
