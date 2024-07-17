# Semantic Kernel is a lightweight, open-source development kit that lets you easily build AI agents and integrate the latest AI models into your C#, Python, or Java codebase.
# It serves as an efficient middleware that enables rapid delivery of enterprise-grade solutions

# https://huggingface.co/google-t5/t5-base
# https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

# SK supports downloading models from the Hugging Face that can perform the following tasks: text-generation, text2text-generation, summarization, and sentence-similarity.
# You can search for models by task at https://huggingface.co/models.

# Semantic Memory is a set of data structures that allow you to store the meaning of text that come from different data sources, and optionally to store the source text too.
# These texts can be from the web, e-mail providers, chats, a database, or from your local directory, and are hooked up to the Semantic Kernel through data source connectors.
# The texts are embedded or compressed into a vector of floats representing mathematically the texts' contents and meaning


# pip install semantic-kernel[hugging_face]==1.2.0, transformers

# TLDR --> Too Long Didn't Read

import asyncio
from services import Service
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.hugging_face import (
    HuggingFaceTextCompletion,
    HuggingFaceTextEmbedding,
)
from semantic_kernel.core_plugins import TextMemoryPlugin
from semantic_kernel.memory import SemanticTextMemory, VolatileMemoryStore
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
    text_service_id = (
        "openai-community/gpt2"  # openai-community/gpt2, google-t5/t5-base
    )
    # text-generation text2text-generation summarization
    kernel.add_service(
        service=HuggingFaceTextCompletion(
            service_id=text_service_id,
            ai_model_id=text_service_id,
            task="text-generation",
        ),
    )
    embed_service_id = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_svc = HuggingFaceTextEmbedding(
        service_id=embed_service_id, ai_model_id=embed_service_id
    )
    kernel.add_service(
        service=embedding_svc,
    )
    memory = SemanticTextMemory(
        storage=VolatileMemoryStore(), embeddings_generator=embedding_svc
    )
    kernel.add_plugin(TextMemoryPlugin(memory), "TextMemoryPlugin")


async def main():
    collection_id = "generic"

    await memory.save_information(
        collection=collection_id, id="info1", text="Sharks are fish."
    )
    await memory.save_information(
        collection=collection_id, id="info2", text="Whales are mammals."
    )
    await memory.save_information(
        collection=collection_id, id="info3", text="Penguins are birds."
    )
    await memory.save_information(
        collection=collection_id, id="info4", text="Dolphins are mammals."
    )
    await memory.save_information(
        collection=collection_id, id="info5", text="Flies are insects."
    )

    # Define prompt function using SK prompt template language
    my_prompt = """I know these animal facts: 
    - {{recall 'fact about sharks'}}
    - {{recall 'fact about whales'}} 
    - {{recall 'fact about penguins'}} 
    - {{recall 'fact about dolphins'}} 
    - {{recall 'fact about flies'}}
    Now, tell me something about: {{$request}}"""

    execution_settings = HuggingFacePromptExecutionSettings(
        service_id=text_service_id,
        ai_model_id=text_service_id,
        max_tokens=50,
        # temperature=0.8,
        # top_p=0.5,
        # do_sample=True,
    )

    prompt_template_config = PromptTemplateConfig(
        template=my_prompt,
        name="text_complete",
        template_format="semantic-kernel",
        execution_settings=execution_settings,
    )

    my_function = kernel.add_function(
        function_name="text_complete",
        plugin_name="TextCompletionPlugin",
        prompt_template_config=prompt_template_config,
    )
    output = await kernel.invoke(
        my_function,
        request="What are whales?",
    )

    output = str(output).strip()

    query_result1 = await memory.search(
        collection=collection_id,
        query="What are whales?",
        limit=1,
        min_relevance_score=0.3,
    )
    print("\n\n")
    print(f"The queried result for 'What are whales?' is: {query_result1[0].text}")

    print(f"{text_service_id} completed prompt with: '{output}'")


# Run the main function
if __name__ == "__main__":
    asyncio.run(main())
