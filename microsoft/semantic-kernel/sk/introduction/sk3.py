# Make sure paths are correct for the imports

import os
import sys
from services import Service
from semantic_kernel import Kernel
from samples.service_settings import ServiceSettings
from semantic_kernel.functions import KernelArguments
import asyncio

notebook_dir = os.path.abspath("")
parent_dir = os.path.dirname(notebook_dir)
grandparent_dir = os.path.dirname(parent_dir)


sys.path.append(grandparent_dir)
from semantic_kernel import Kernel

kernel = Kernel()


service_settings = ServiceSettings()

# Select a service to use for this notebook (available services: OpenAI, AzureOpenAI, HuggingFace)
selectedService = (
    Service.AzureOpenAI
    if service_settings.global_llm_service is None
    else Service(service_settings.global_llm_service.lower())
)
print(f"Using service type: {selectedService}")


# Remove all services so that this cell can be re-run without restarting the kernel
kernel.remove_all_services()

service_id = None
if selectedService == Service.OpenAI:
    from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

    service_id = "default"
    kernel.add_service(
        OpenAIChatCompletion(
            service_id=service_id,
        ),
    )
elif selectedService == Service.AzureOpenAI:
    from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion

    service_id = "default"
    kernel.add_service(
        AzureChatCompletion(
            service_id=service_id,
        ),
    )


plugin = kernel.add_plugin(
    parent_directory="../../prompt_template_samples/", plugin_name="FunPlugin"
)


# Run your prompt
# Note: functions are run asynchronously
async def main():
    joke_function = plugin["Joke"]

    joke = await kernel.invoke(
        joke_function,
        KernelArguments(
            input="time travel to stone age dressing a astronaut suit",
            style="super silly",
        ),
    )
    print(joke)


if __name__ == "__main__":
    asyncio.run(main())
