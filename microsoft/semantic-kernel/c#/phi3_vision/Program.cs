// LLMOps:Microsoft Phi-3 Vision 128k Instruct Muti-Modal Prompts in CPU with ONNX Runtime in C#
//
// dotnet add package Spectre.Console --version 0.49.1 A .NET library that makes it easier to create beautiful, cross platform, console applications.
// dotnet add package Microsoft.ML.OnnxRuntimeGenAI --version 0.3.0  This API gives you an easy, flexible and performant way of running LLMs on device.

// ONNX (Open Neural Network Exchange) 
// huggingface-cli download microsoft/Phi-3-vision-128k-instruct-onnx-cpu --include cpu-int4-rtn-block-32-acc-level-4/* --local-dir .
// https://onnxruntime.ai/docs/genai/tutorials/phi3-python.html
// pip install huggingface-hub[cli]

// Phi-3-vision-128k-instruct allows Phi-3 to not only understand language, but also see the world visually. Through Phi-3-vision-128k-instruct,
// we can solve different visual problems, such as OCR, table analysis, object recognition, describe the picture etc.
// We can easily complete tasks that previously required a lot of data training. The following are related techniques and application scenarios cited by
// Phi-3-vision-128k-instruct

using Microsoft.ML.OnnxRuntimeGenAI;
using Phi3VisionOnnxConsole.Utils;
using Spectre.Console;

// Show the header
ConsoleHelper.ShowHeader();

// Get the model path from the user
string modelPath
    = ConsoleHelper.GetFolderPath(Statics.ModelInputPrompt);

// Show the header
ConsoleHelper.ShowHeader();
ConsoleHelper.WriteToConsole(Statics.ModelLoadingMessage);

// Load the model and tokenizer
using Model model = new(modelPath);
using MultiModalProcessor processor = new(model);
using Tokenizer tokenizer = new(model);

// Show the header
ConsoleHelper.ShowHeader();

// Simulate the chat loop
while (true)
{
    // Get path to a picture file
    string picturePath
        = ConsoleHelper.GetFilePath(Statics.PictureInputPrompt);

    // Load the image
    Images image =
        Images.Load(picturePath);

    // Show the header
    ConsoleHelper.ShowHeader();

    // Show process message
    ConsoleHelper.WriteToConsole(Statics.AnalyzeImageMessage);


    string prompt_User = ConsoleHelper.Getprompt(Statics.UserPrompt);
    // Show the header
    ConsoleHelper.ShowHeader();
    // Show process message
    ConsoleHelper.WriteToConsole(Statics.Execute_Task);
    ConsoleHelper.WriteToConsole(Environment.NewLine);
    ConsoleHelper.WriteToConsole(Environment.NewLine);
    ConsoleHelper.WriteToConsole(Statics.OutputPrompt);

    // Create the prompt Original Statics.UserImagePrompt
    string fullPrompt
        = $"<|system|>{Statics.SystemPrompt}<|end|>" +
          $"<|user|><|image_1|>{prompt_User}<|end|>" +
          $"<|assistant|>";

    // Process the image
    NamedTensors inputTensors
        = processor.ProcessImages(fullPrompt, image);

    // Specify the generator parameters
    using GeneratorParams generatorParams = new(model);
    generatorParams.SetSearchOption("max_length", 4096);
    generatorParams.SetInputs(inputTensors);

    // Create the generator
    using Generator generator = new(model, generatorParams);
    DateTime currentDateTime1 = DateTime.Now;
    // Generate the response
    while (!generator.IsDone())
    {
        generator.ComputeLogits();
        generator.GenerateNextToken();

        string output = tokenizer.Decode(generator.GetSequence(0)[^1..]);

        // Break if the end token is found
        if (output.Contains("</s>"))
        {
            break;
        }

        ConsoleHelper.WriteToConsole(output);

    }
    DateTime currentDateTime2 = DateTime.Now;
    string diffttime = currentDateTime2.Subtract(currentDateTime1).TotalMinutes.ToString();
    ConsoleHelper.WriteToConsole(Environment.NewLine);
    ConsoleHelper.WriteToConsole(Environment.NewLine);
    ConsoleHelper.WriteToConsole(diffttime);
    ConsoleHelper.WriteToConsole(Environment.NewLine);
    ConsoleHelper.WriteToConsole(Statics.RestartPrompt);

    // Wait for the user to press a key
    var key =  Console.ReadKey();
    if (key.Key == ConsoleKey.Escape)
    {
        break;
    }
}
