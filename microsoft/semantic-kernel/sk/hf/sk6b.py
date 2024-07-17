# https://huggingface.co/google-t5/t5-base
# https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

# SK supports downloading models from the Hugging Face that can perform the following tasks: text-generation, text2text-generation, summarization, and sentence-similarity.
# You can search for models by task at https://huggingface.co/models.

import asyncio
from services import Service
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.hugging_face import (
    HuggingFaceTextCompletion,
)

from semantic_kernel.connectors.ai.hugging_face import (
    HuggingFacePromptExecutionSettings,
)
from semantic_kernel.prompt_template import PromptTemplateConfig
from dotenv import load_dotenv
import logging
import transformers


logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
logging.getLogger("semantic_kernel").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

load_dotenv()

kernel = Kernel()

# Select a service to use (available services: OpenAI, AzureOpenAI, HuggingFace)
selectedService = Service.HuggingFace
print(f"Using service type: {selectedService}")

# Configure LLM service
if selectedService == Service.HuggingFace:
    # Feel free to update this model to any other model available on Hugging Face
    text_service_id = "google-t5/t5-base"
    kernel.add_service(
        service=HuggingFaceTextCompletion(
            service_id=text_service_id,
            ai_model_id=text_service_id,
            task="summarization",
        ),
    )
    prompt = "{{$input}}\n\nOne line TLDR with the fewest words."
    execution_settings = HuggingFacePromptExecutionSettings(
        service_id=text_service_id,
        ai_model_id=text_service_id,
        max_tokens=30,
        # temperature=0.5,
        # top_p=0.5,
        # do_sample=True,
    )

    prompt_template_config = PromptTemplateConfig(
        template=prompt,
        name="text_complete",
        template_format="semantic-kernel",
        execution_settings=execution_settings,
    )

    summarize = kernel.add_function(
        function_name="text_complete",
        plugin_name="tldr_plugin",
        prompt_template_config=prompt_template_config,
    )


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


# Run the main function
if __name__ == "__main__":
    asyncio.run(main())
