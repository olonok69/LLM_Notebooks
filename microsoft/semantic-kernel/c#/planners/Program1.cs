// AI PLANNERS

// Semantic Kernel SDK supports planners, which use artificial intelligence (AI) to automatically call the appropriate plugins for a given scenario.

// dotnet add package Microsoft.SemanticKernel.Planners.Handlebars --version 1.2.0-preview    NOTE PREVIEW
// dotnet add package Microsoft.SemanticKernel


using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Planning.Handlebars;
using Microsoft.SemanticKernel.PromptTemplates.Handlebars;
using System.ComponentModel;

#pragma warning disable CS8600, CS8604, SKEXP0060
string yourDeploymentName = "gpt-4o-chat";
string yourEndpoint = "https://open-ai-olonok.openai.azure.com";
string yourApiKey = Environment.GetEnvironmentVariable("AZURE_OPENAI_API_KEY");

// FIRST EXAMPLE

var builder = Kernel.CreateBuilder();
builder.AddAzureOpenAIChatCompletion(
    yourDeploymentName,
    yourEndpoint,
    yourApiKey,
    "gpt4o");
var kernel = builder.Build();
kernel.ImportPluginFromType<MusicLibraryPlugin>();
kernel.ImportPluginFromType<MusicConcertPlugin>();
kernel.ImportPluginFromPromptDirectory("D:\\repos2\\c#\\planners\\planners\\Prompts\\");


var planner = new HandlebarsPlanner(new HandlebarsPlannerOptions() { AllowLoops = true });

string location = "Redmond WA USA";
string goal = @$"Based on the user's recently played music, suggest a 
    concert for the user living in ${location}";

var concertPlan = await planner.CreatePlanAsync(kernel, goal);
// output the plan result
Console.WriteLine("Concert Plan:");
Console.WriteLine(concertPlan);
Console.WriteLine("\nPress any key to continue...\n\n");
Console.ReadKey();

var songSuggesterFunction = kernel.CreateFunctionFromPrompt(
    promptTemplate: @"Based on the user's recently played music:
        {{$recentlyPlayedSongs}}
        recommend a song to the user from the music library:
        {{$musicLibrary}}",
    functionName: "SuggestSong",
    description: "Suggest a song to the user"
);

kernel.Plugins.AddFromFunctions("SuggestSongPlugin", [songSuggesterFunction]);

var songSuggestPlan = await planner.CreatePlanAsync(kernel, @"Suggest a song from the 
    music library to the user based on their recently played songs");

Console.WriteLine("Song Plan:");
Console.WriteLine(songSuggestPlan);

Console.WriteLine("\nPress any key to continue...\n\n");
Console.ReadKey();


