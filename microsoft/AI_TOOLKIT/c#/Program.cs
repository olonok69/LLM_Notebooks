// dotnet add package Microsoft.ML.OnnxRuntimeGenAI.DirectML --version 0.3.0
//dotnet add package Microsoft.ML.OnnxRuntimeGenAI.Cuda --version 0.3.0
// dotnet add package Microsoft.ML.OnnxRuntimeGenAI --version 0.3.0


using Microsoft.ML.OnnxRuntimeGenAI ;

// update user_name and path placeholders
string modelPath = "C:\\Users\\User\\.aitk\\models\\microsoft\\Phi-3-mini-128k-instruct-onnx\\cpu_and_mobile\\cpu-int4-rtn-block-32-acc-level-4";
Console.Write("Loading model from " + modelPath + "...");
using Model model = new(modelPath);
Console.Write("Done\n");
using Tokenizer tokenizer = new(model);
using TokenizerStream tokenizerStream = tokenizer.CreateStream();

while (true)
{
    Console.Write("User:");

    string? input = Console.ReadLine();
    string prompt = "<|user|>\n" + input + "<|end|>\n<|assistant|>";

    var sequences = tokenizer.Encode(prompt);

    using GeneratorParams generatorParams = new GeneratorParams(model);
    generatorParams.SetSearchOption("max_length", 512);
    generatorParams.SetInputSequences(sequences);

    Console.Out.Write("\nAI:");
    using Generator generator = new(model, generatorParams);
    while (!generator.IsDone())
    {
        generator.ComputeLogits();
        generator.GenerateNextToken();
        Console.Out.Write(tokenizerStream.Decode(generator.GetSequence(0)[^1]));
        Console.Out.Flush();
    }
    Console.WriteLine();
}
