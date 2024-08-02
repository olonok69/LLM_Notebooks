// Smantic Kernel and Chroma Vector store

// Chroma
/* https://docs.trychroma.com/
 * 
 * 
 * Chroma is an open-source embedding database designed to make it easy to build Language Model applications by making knowledge, facts, and plugins 
 * pluggable for LLMs. It allows us to store and retrieve information in a way that can be easily utilized by the models, enabling both short-term and long-term
 * memory for more advanced applications. 
 
Install
docker pull chromadb/chroma
docker run -it --rm -p 8000:8000/tcp chromadb/chroma:latest
 * */

/* INSTALL
dotnet add package Microsoft.SemanticKernel
dotnet add package Microsoft.SemanticKernel.Plugins.Memory --version 1.16.2-alpha
dotnet add package Microsoft.SemanticKernel.Connectors.Chroma --version 1.16.2-alpha
dotnet add package System.Linq.Async
*/


using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Connectors.Chroma;
using Microsoft.SemanticKernel.Connectors.OpenAI;
using Microsoft.SemanticKernel.Memory;
using Microsoft.SemanticKernel.Plugins.Memory;
using Kernel = Microsoft.SemanticKernel.Kernel;

#pragma warning disable CS8600, CS8604, SKEXP0060
string yourDeploymentName = "gpt-4o-chat";
string yourEndpoint = "https://open-ai-olonok.openai.azure.com";
string yourApiKey = Environment.GetEnvironmentVariable("AZURE_OPENAI_API_KEY");

var builder = Kernel.CreateBuilder();
builder.AddAzureOpenAIChatCompletion(
    yourDeploymentName,
    yourEndpoint,
    yourApiKey,
    "gpt4o");

var kernel = builder.Build();

#pragma warning disable SKEXP0001, SKEXP0010, SKEXP0020, SKEXP0050



var memoryBuilder = new MemoryBuilder();


memoryBuilder.WithAzureOpenAITextEmbeddingGeneration("text-embedding-ada-002", yourEndpoint, yourApiKey, "model-id");


var chromaMemoryStore = new ChromaMemoryStore("http://127.0.0.1:8000");

memoryBuilder.WithMemoryStore(chromaMemoryStore);

var memory = memoryBuilder.Build();


const string MemoryCollectionName = "aboutMe";

/*await memory.SaveInformationAsync(MemoryCollectionName, id: "info1", text: "My name is Andrea");
await memory.SaveInformationAsync(MemoryCollectionName, id: "info2", text: "I currently work as a tourist operator");
await memory.SaveInformationAsync(MemoryCollectionName, id: "info3", text: "I currently live in Seattle and have been living there since 2005");
await memory.SaveInformationAsync(MemoryCollectionName, id: "info4", text: "I visited France and Italy five times since 2015");
await memory.SaveInformationAsync(MemoryCollectionName, id: "info5", text: "My family is from New York");*/

//await memory.SaveInformationAsync(MemoryCollectionName, id: "info6", text: "I have also family in Cuba");

//await memory.SaveInformationAsync(MemoryCollectionName, id: "info7", text: "My Hobbies are Music and diving");
/*
var questions = new[]
{
    "what is my name?",
    "where do I live?",
    "where is my family from?",
    "where have I travelled?",
    "what do I do for work?",
     "where also I have family?",
     "what are my hobbies"
};

foreach (var q in questions)
{
    var response = await memory.SearchAsync(MemoryCollectionName, q, limit: 1, minRelevanceScore: 0.5).FirstOrDefaultAsync();
    Console.WriteLine(q + " " + response?.Metadata.Text);
}*/


#pragma warning disable SKEXP0001, SKEXP0050

// TextMemoryPlugin provides the "recall" function
kernel.ImportPluginFromObject(new TextMemoryPlugin(memory));


const string skPrompt = @"
ChatBot can have a conversation with you about any topic.
It can give explicit instructions or say 'I don't know' if it does not have an answer.

Information about me, from previous conversations:
- {{$fact1}} {{recall $fact1}}
- {{$fact2}} {{recall $fact2}}
- {{$fact3}} {{recall $fact3}}
- {{$fact4}} {{recall $fact4}}
- {{$fact5}} {{recall $fact5}}
- {{$fact6}} {{recall $fact6}}
- {{$fact7}} {{recall $fact7}}
Chat:
{{$history}}
User: {{$userInput}}
ChatBot: ";

var chatFunction = kernel.CreateFunctionFromPrompt(skPrompt, new OpenAIPromptExecutionSettings { MaxTokens = 400, Temperature = 0.8 });


#pragma warning disable SKEXP0001, SKEXP0050

var arguments = new KernelArguments();

arguments["fact1"] = "what is my name?";
arguments["fact2"] = "where do I live?";
arguments["fact3"] = "where is my family from?";
arguments["fact4"] = "where have I travelled? consider my hobbies";
arguments["fact5"] = "what do I do for work?";
arguments["fact6"] = "where also I have family?";
arguments["fact7"] = "what are my hobbies?";

arguments[TextMemoryPlugin.CollectionParam] = MemoryCollectionName;
arguments[TextMemoryPlugin.LimitParam] = "2";
arguments[TextMemoryPlugin.RelevanceParam] = "0.5";

var history = "";
arguments["history"] = history;
Func<string, Task> Chat = async (string input) => {
    // Save new message in the kernel arguments
    arguments["userInput"] = input;

    // Process the user message and get an answer
    var answer = await chatFunction.InvokeAsync(kernel, arguments);

    // Append the new interaction to the chat history
    var result = $"\nUser: {input}\nChatBot: {answer}\n";

    history += result;
    arguments["history"] = history;

    // Show the bot response
    Console.WriteLine(result);
};

await Chat("Hello, I think we've met before, remember? my name is...");
Console.WriteLine("\nPress any key to continue...\n\n");
Console.ReadKey();

await Chat("what are my hobbies");
Console.WriteLine("\nPress any key to continue...\n\n");
Console.ReadKey();

await Chat("I want to plan a trip and visit my family. Do you know where that is?");
Console.WriteLine("\nPress any key to continue...\n\n");
Console.ReadKey();


await Chat("Great! What are some fun things to do there?");

