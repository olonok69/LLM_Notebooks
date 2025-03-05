from vllm import LLM, SamplingParams

# In this script, we demonstrate how to pass input to the chat method:
conversation = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hello! How can I assist you today?"},
    {
        "role": "user",
        "content": "Write an essay about the importance of higher education.",
    },
]

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=512)

# Create an LLM.
llm = LLM(
    model="/mnt/wolverine/home/samtukra/juan/models/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_L.gguf",
    tokenizer="/mnt/wolverine/home/samtukra/juan/models/Meta-Llama-3.1-8B-Instruct-GGUF",
    max_model_len=50000,
    quantization="int8",
    device="cuda",
)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.chat(conversation, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print("\n")
    print(f"Prompt: {prompt!r},\n\n Generated text: {generated_text!r}")
