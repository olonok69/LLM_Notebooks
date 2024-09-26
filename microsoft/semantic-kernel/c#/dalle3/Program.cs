/* dotnet add package Microsoft.SemanticKernel --version 1.19.0
 https://github.com/microsoft/semantic-kernel/tree/main
  Semantic Kernel is an SDK that integrates Large Language Models (LLMs) like OpenAI, Azure OpenAI, and Hugging Face with conventional programming languages 
  like C#, Python, and Java. Semantic Kernel achieves this by allowing you to define plugins that can be chained together in just a few lines of code.
  
dotnet add package System.Numerics.Tensors  --version 8.0.0

dotnet add package  SkiaSharp --version 2.88.3
https://github.com/mono/SkiaSharp
SkiaSharp is a cross-platform 2D graphics API for .NET platforms based on Google's Skia Graphics Library (skia.org). 
It provides a comprehensive 2D API that can be used across mobile, server and desktop models to render images.

dotnet add package Spectre.Console --version 0.49.1

Spectre.Console is a .NET library that makes it easier to create beautiful console applications.
https://spectreconsole.net/#:~:text=Spectre.Console.Cli.%20Create%20strongly%20typed%20settings%20and

*/


using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.TextToImage;
using Microsoft.SemanticKernel.Embeddings;
using Microsoft.SemanticKernel.Connectors.OpenAI;
using System.Numerics.Tensors;
using classification.config;
using classification.ConsoleDemo.Utils;
using Spectre.Console;



#pragma warning disable SKEXP0001

public static class Programm
{
    static async Task Main(string[] args)
    {

        ConsoleHelper.ShowHeader();
        ConsoleHelper.WriteToConsole(Environment.NewLine);

        var builder = Kernel.CreateBuilder();

        var azureopenaikey = Environment.GetEnvironmentVariable("AZURE_OPENAI_API_KEY");
         #pragma warning disable SKEXP0001, SKEXP0010
        builder.AddAzureOpenAIChatCompletion(
                 "gpt-3-5-16k",                      // Azure OpenAI Deployment Name
                 "https://open-ai-olonok.openai.azure.com", // Azure OpenAI Endpoint
                 azureopenaikey);

        builder.AddAzureOpenAITextEmbeddingGeneration("text-embedding-ada-002", "https://open-ai-olonok.openai.azure.com", azureopenaikey);

        builder.AddAzureOpenAITextToImage("Dalle3", "https://open-ai-olonok.openai.azure.com", azureopenaikey);

        var kernel = builder.Build();

        // Get AI service instance used to generate images
        var dallE = kernel.GetRequiredService<ITextToImageService>();

        // Get AI service instance used to extract embedding from a text
        var textEmbedding = kernel.GetRequiredService<ITextEmbeddingGenerationService>();
        while (true)

        {

            var prompt = @"
            Think about an artificial object correlated to number {{$input}}.
            Describe the image with one detailed sentence. The description cannot contain numbers.";

            var executionSettings = new OpenAIPromptExecutionSettings
            {
                MaxTokens = 256,
                Temperature = 1
            };

            // Create a semantic function that generate a random image description.
            var genImgDescription = kernel.CreateFunctionFromPrompt(prompt, executionSettings);

            string random = ConsoleHelper.GetOutputPath(Statics.TaskDalle);
            ConsoleHelper.WriteToConsole(Environment.NewLine);
            var imageDescriptionResult = await kernel.InvokeAsync(genImgDescription, new() { ["input"] = random });
            var imageDescription = imageDescriptionResult.ToString();

            // Use DALL-E 3 to generate an image. OpenAI in this case returns a URL (though you can ask to return a base64 image)
            var imageUrl = await dallE.GenerateImageAsync(imageDescription.Trim(), 1024, 1024);

            string outPath = ConsoleHelper.GetOutputPath(Statics.ImageOutputPath);
            ConsoleHelper.WriteToConsole(Environment.NewLine);

            await SkiaUtils.SaveImage(imageUrl, 1024, 1024, outPath);
            // provide
            string guess = ConsoleHelper.GetOutputPath(Statics.IntroduceGuess);
            ConsoleHelper.WriteToConsole(Environment.NewLine);

            // Compare user guess with real description and calculate score
            var origEmbedding = await textEmbedding.GenerateEmbeddingsAsync(new List<string> { imageDescription });
            var guessEmbedding = await textEmbedding.GenerateEmbeddingsAsync(new List<string> { guess });
            var similarity = TensorPrimitives.CosineSimilarity(origEmbedding.First().Span, guessEmbedding.First().Span);
            ConsoleHelper.WriteToConsole($"Your description:\n{Utils.WordWrap(guess, 90)}\n");
            ConsoleHelper.WriteToConsole(Environment.NewLine);
            ConsoleHelper.WriteToConsole($"Real description:\n{Utils.WordWrap(imageDescription.Trim(), 90)}\n");
            ConsoleHelper.WriteToConsole(Environment.NewLine);
            ConsoleHelper.WriteToConsole($"Score: {similarity:0.00}\n\n");
            ConsoleHelper.WriteToConsole(Environment.NewLine);
            var key = Console.ReadKey();
            if (key.Key == ConsoleKey.Escape)
            {
                break;
            }
            else
            {
                ConsoleHelper.ShowHeader();
            }

        }
    }
}



