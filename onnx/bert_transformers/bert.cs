// packages 
// dotnet add package Microsoft.ML.OnnxRuntime --version 1.16.0
// dotnet add package Microsoft.ML.OnnxRuntime.Managed --version 1.16.0
// dotnet add package Microsoft.ML
// dotnet add package BERTTokenizers --version 1.1.0


using BERTTokenizers;
using Microsoft.ML.Data;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;

internal class BertTokenizeProgram
{
    static void Main(string[] args)

    {
        if (args.Length == 0)
        {
            System.Console.WriteLine("Please enter a numeric argument.");
            
        }
        var question =  args[0].ToString();
        var context = args[1].ToString();

        // var sentence = "{\"question\": \"Where is Bob Dylan From?\", \"context\": \"Bob Dylan is from Duluth, Minnesota and is an American singer-songwriter\"}";
        var sentence = "{\"question\": "+ question + " \"context\": " + context +" }";
        Console.WriteLine(sentence);
        // Create Tokenizer and tokenize the sentence.
        var tokenizer = new BertUncasedLargeTokenizer();

        // Get the sentence tokens.
        var tokens = tokenizer.Tokenize(sentence);
        // Console.WriteLine(String.Join(", ", tokens));

        // Encode the sentence and pass in the count of the tokens in the sentence.
        var encoded = tokenizer.Encode(tokens.Count(), sentence);

        // Break out encoding to InputIds, AttentionMask and TypeIds from list of (input_id, attention_mask, type_id).
        var bertInput = new BertInput()
        {
            InputIds = encoded.Select(t => t.InputIds).ToArray(),
            AttentionMask = encoded.Select(t => t.AttentionMask).ToArray(),
            TypeIds = encoded.Select(t => t.TokenTypeIds).ToArray(),
        };


        // Get path to model to create inference session.
        var modelPath = @"d:\repos\onnx\models\bert-large-uncased-whole-word-masking-finetuned-squad-17.onnx";
        // using var gpuSessionOptions = SessionOptions.MakeSessionOptionWithCudaProvider(0);
        using var runOptions = new RunOptions();
        using var session = new InferenceSession(modelPath); //gpuSessionOptions

        // Create input tensors over the input data.
        using var inputIdsOrtValue = OrtValue.CreateTensorValueFromMemory(bertInput.InputIds,
              new long[] { 1, bertInput.InputIds.Length });

        using var attMaskOrtValue = OrtValue.CreateTensorValueFromMemory(bertInput.AttentionMask,
              new long[] { 1, bertInput.AttentionMask.Length });

        using var typeIdsOrtValue = OrtValue.CreateTensorValueFromMemory(bertInput.TypeIds,
              new long[] { 1, bertInput.TypeIds.Length });

        // Create input data for session. Request all outputs in this case.
        var inputs = new Dictionary<string, OrtValue>
  {
      { "input_ids", inputIdsOrtValue },
      { "input_mask", attMaskOrtValue },
      { "segment_ids", typeIdsOrtValue }
  };


        // Run session and send the input data in to get inference output. 
        using var output = session.Run(runOptions, inputs, session.OutputNames);

        // Get the Index of the Max value from the output lists.
        // We intentionally do not copy to an array or to a list to employ algorithms.
        // Hopefully, more algos will be available in the future for spans.
        // so we can directly read from native memory and do not duplicate data that
        // can be large for some models
        // Local function
        int GetMaxValueIndex(ReadOnlySpan<float> span)
        {
            float maxVal = span[0];
            int maxIndex = 0;
            for (int i = 1; i < span.Length; ++i)
            {
                var v = span[i];
                if (v > maxVal)
                {
                    maxVal = v;
                    maxIndex = i;
                }
            }
            return maxIndex;
        }

        var startLogits = output[0].GetTensorDataAsSpan<float>();
        int startIndex = GetMaxValueIndex(startLogits);

        var endLogits = output[output.Count - 1].GetTensorDataAsSpan<float>();
        int endIndex = GetMaxValueIndex(endLogits);

        var predictedTokens = tokens
                      .Skip(startIndex)
                      .Take(endIndex + 1 - startIndex)
                      .Select(o => tokenizer.IdToToken((int)o.VocabularyIndex))
                      .ToList();

        // Print the result.
        Console.WriteLine(String.Join(" ", predictedTokens));
        Console.WriteLine("Press any key to continue...");
        Console.ReadKey();

    }
}


    public struct BertInput
    {
        public long[] InputIds { get; set; }
        public long[] AttentionMask { get; set; }
        public long[] TypeIds { get; set; }
    }


