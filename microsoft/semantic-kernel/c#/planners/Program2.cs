// AI PLANNERS

// Semantic Kernel SDK supports planners, which use artificial intelligence (AI) to automatically call the appropriate plugins for a given scenario.

// dotnet add package Microsoft.SemanticKernel.Planners.Handlebars --version 1.2.0-preview    NOTE PREVIEW
// dotnet add package Microsoft.SemanticKernel

// Performance
// You might notice that to run a plan, you must first call CreatePlanAsync and then invoke the plan. It takes time for the planner to consume a full list of
// tokens and generate a plan for the goal. It also takes time to run the plan. If you rely only on the planner's response after a user provides input,
// they might assume the application is unresponsive. You might want to consider providing some feedback or some initial response from the model while the user
// waits.

//Cost
// Another consideration is cost. Prompts and generated plans can consume a significant number of tokens. This token consumption can result in high costs for
// your service if you're not careful, especially since planning typically requires your model to support higher token counts. You may want to use predefined
// plans for common scenarios, or adjust your functions to support fewer tokens.

//Correctness
//It's possible for planners to generate faulty plans. For example, they may pass variables incorrectly, return malformed schemas, or perform steps that
//don't make sense. Some errors can be recovered by asking planner to "fix" the plan. Implementing robust error handling can help you ensure your planner
//is reliable.

// Another way to ensure reliability is to use pregenerated plans. There may be common scenarios that your users frequently ask for.
// To avoid the performance hit and costs associated with a planner, you can predefine plans and serve them up to a user. To predefine your plans, you can
// generate plans for common scenarios offline and store them within your project. Based on the intent of the user, you can then serve the plan you created
// previously. Using predefined plans for common scenarios is a great strategy to improve performance, reliability, and reduce costs.


using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Planning.Handlebars;
using Microsoft.SemanticKernel.PromptTemplates.Handlebars;
using System.ComponentModel;

#pragma warning disable CS8600, CS8604, SKEXP0060
string yourDeploymentName = "gpt-4o-chat";
string yourEndpoint = "https://open-ai-olonok.openai.azure.com";
string yourApiKey = Environment.GetEnvironmentVariable("AZURE_OPENAI_API_KEY");

// Second EXAMPLE

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


var songSuggesterFunction = kernel.CreateFunctionFromPrompt(
    promptTemplate: @"Based on the user's recently played music:
    {{$recentlyPlayedSongs}}
    recommend a song to the user from the music library:
    {{$musicLibrary}}",
    functionName: "SuggestSong",
    description: "Suggest a song to the user"
);

kernel.Plugins.AddFromFunctions("SuggestSongPlugin", [songSuggesterFunction]);


string dir = Directory.GetCurrentDirectory();
string template = File.ReadAllText("D:\\repos2\\c#\\planners\\planners\\handlebarsTemplate.txt");

var handlebarsPromptFunction = kernel.CreateFunctionFromPrompt(
    new()
    {
        Template = template,
        TemplateFormat = "handlebars"
    }, new HandlebarsPromptTemplateFactory()
);

string location = "Redmond WA USA";

var templateResult = await kernel.InvokeAsync(handlebarsPromptFunction,
    new() {
        { "location", location },
        { "suggestConcert", false }
    });

Console.WriteLine(templateResult);
Console.WriteLine("\nPress any key to continue...\n\n");
Console.ReadKey();