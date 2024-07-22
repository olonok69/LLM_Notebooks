//dotnet add package Microsoft.SemanticKernel
// https://github.com/microsoft/SemanticKernelCookBook
// dotnet add package OllamaSharp --version 2.0.10
// https://github.com/awaescher/OllamaSharp
// OLLAMA
//  docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
// ollama 


#pragma warning disable SKEXP0010
using Microsoft.SemanticKernel;

namespace ollamasharp;
class Program
{
    async static Task Main(string[] args)
    {
        if (args.Length == 0)
        {
            Console.WriteLine("Please enter a question to send to the CoT Bot.");

        }
        string question = args[0].ToString();
       

        var endpoint = new Uri("http://localhost:11434");
        var modelId = "phi3";

        var kernelBuilder = Kernel.CreateBuilder().AddOpenAIChatCompletion(modelId: modelId, apiKey: null, endpoint: endpoint);
        var kernel = kernelBuilder.Build();


        const string skPrompt = @"
        Read the Instruction below and provide an answer.
        Think step by step and provide: the final response and a short the rational of your answer. No more than 3 lines

        {{$history}}
        User: {{$userInput}}
        ChatBot:";

        var chatFunction = kernel.CreateFunctionFromPrompt(skPrompt);
        var history = "";
        var arguments = new KernelArguments()
        {
            ["history"] = history
        };


        //var userInput = "Hi, I'm looking for book suggestions";
        arguments["userInput"] = question;

        var bot_answer =  await chatFunction.InvokeAsync(kernel, arguments);


        history += $"\nUser: {question}\nAI: {bot_answer}\n";
        arguments["history"] = history;

        Console.WriteLine(history);


    }
}