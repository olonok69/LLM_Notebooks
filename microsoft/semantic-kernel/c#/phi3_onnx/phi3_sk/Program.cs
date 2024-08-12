// dotnet add package Microsoft.ML.OnnxRuntime --version 1.18.0
// dotnet add package Spectre.Console --version 0.49.1 A .NET library that makes it easier to create beautiful, cross platform, console applications.
// dotnet add package Microsoft.ML.OnnxRuntimeGenAI --version 0.3.0  This API gives you an easy, flexible and performant way of running LLMs on device.
// dotnet add package Microsoft.ML.OnnxRuntimeGenAI.Cuda --version 0.3.0
// dotnet add package Microsoft.SemanticKernel
// dotnet add package feiyun0112.SemanticKernel.Connectors.OnnxRuntimeGenAI.CPU or feiyun0112.SemanticKernel.Connectors.OnnxRuntimeGenAI.CUDA
// https://github.com/feiyun0112/SemanticKernel.Connectors.OnnxRuntimeGenAI/tree/main


// Download phi3 onnx
// https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx
// https://onnxruntime.ai/docs/genai/tutorials/phi3-python.html
// pip install huggingface-hub[cli]

// huggingface-cli download microsoft/Phi-3-mini-4k-instruct-onnx --include cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/* --local-dir .



using feiyun0112.SemanticKernel.Connectors.OnnxRuntimeGenAI;
using Microsoft.SemanticKernel;
using Phi3OnnxConsole.Utils;
using Spectre.Console;


// Show the header
ConsoleHelper.ShowHeader();
// Get the model path from the user
string modelPath
    = ConsoleHelper.GetFolderPath(Statics.ModelInputPrompt);

// Show the header
ConsoleHelper.ShowHeader();
ConsoleHelper.WriteToConsole(Statics.ModelLoadingMessage);

Kernel kernel = Kernel.CreateBuilder()
           .AddOnnxRuntimeGenAIChatCompletion(
               modelPath: modelPath)
           .Build();
var systemPrompt = "You are an AI assistant that helps people find information. Answer questions using a direct style. Do not share more information that the requested by the users.";

// chat start
// Show process message


var key2 = Console.ReadKey();
// chat loop
while (true)
{
    // Get user question
    ConsoleHelper.WriteToConsole(Environment.NewLine);
    var userQ = ConsoleHelper.Getprompt(Statics.UserPrompt);
    ConsoleHelper.ShowHeader();
    ConsoleHelper.WriteToConsole(Statics.Execute_Task);
    ConsoleHelper.WriteToConsole(Environment.NewLine);

    if (string.IsNullOrEmpty(userQ))
    {
        break;
    }

    // show phi3 response
    ConsoleHelper.WriteToConsole(Statics.PhiChat);
    var fullPrompt = $"<|system|>{systemPrompt}<|end|><|user|>{userQ}<|end|><|assistant|>";
    float temperature = 0.9f;
    await foreach (string text in kernel.InvokePromptStreamingAsync<string>(fullPrompt,
                   new KernelArguments(new OnnxRuntimeGenAIPromptExecutionSettings() { MaxLength = 2048, Temperature= temperature })))
    {
        if (text.Contains("</s>"))
        {
            break;
        }
        ConsoleHelper.WriteToConsole(text);
    }
    ConsoleHelper.WriteToConsole(Environment.NewLine);
    ConsoleHelper.WriteToConsole(Statics.RestartPrompt);
    ConsoleHelper.WriteToConsole(Environment.NewLine);
    var key = Console.ReadKey();
    if (key.Key == ConsoleKey.Escape)
    {
        break;
    }
}