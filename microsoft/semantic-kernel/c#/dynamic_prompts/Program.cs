// Install
//dotnet add package Microsoft.SemanticKernel


// The Semantic Kernel SDK allows developers to run functions within prompts to create intelligent applications.
// Functions nested within your prompts can perform a wide range of tasks to make your AI agent more robust.
// This allows you to perform tasks that large language models can't typically complete on their own.
// Using variables.
// Calling external functions.
// Passing arguments to functions.


// https://learn.microsoft.com/en-us/semantic-kernel/concepts/plugins/adding-native-plugins?pivots=programming-language-csharp
// https://learn.microsoft.com/en-us/semantic-kernel/concepts/plugins/?pivots=programming-language-csharp


using System;
using Microsoft.SemanticKernel;
using System.ComponentModel;
using System.Text.Json;
using System.Text.Json.Nodes;
using native_function;
#pragma warning disable SKEXP0010

var builder = Kernel.CreateBuilder();

// var azureopenaikey = Environment.GetEnvironmentVariable("AZURE_OPENAI_API_KEY");

//builder.AddAzureOpenAIChatCompletion(
//         "gpt-3-5-16k",                      // Azure OpenAI Deployment Name
//         "https://open-ai-olonok.openai.azure.com", // Azure OpenAI Endpoint
//         azureopenaikey);


var endpoint = new Uri("http://localhost:11434");
var modelId = "phi3";

var kernelBuilder = Kernel.CreateBuilder().AddOpenAIChatCompletion(modelId: modelId, apiKey: null, endpoint: endpoint);
var kernel = kernelBuilder.Build();

kernel.ImportPluginFromType<MusicLibraryPlugin>();

// add new song to the list of recently played
var result = await kernel.InvokeAsync(
    "MusicLibraryPlugin",
    "AddToRecentlyPlayed",
    new()
    {
        ["artist"] = "Evanescence",
        ["song"] = "you are not the only one",
        ["genre"] = "Techno, glam, rock"
    }
);

Console.WriteLine(result);
// Send prompt to model 

string prompt = @"This is a list of music available to the user:
    {{MusicLibraryPlugin.GetMusicLibrary }} 
  
    This is a list of music the user has recently played:
    {{MusicLibraryPlugin.GetRecentPlays}}
    
    Based on their recently played music, suggest a song from
    the list to play next";

var result2 = await kernel.InvokePromptAsync(prompt);
Console.WriteLine(result2);