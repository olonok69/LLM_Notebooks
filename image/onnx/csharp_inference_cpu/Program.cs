//
// Run  pretrained  ONNX model using the Onnx Runtime C# API.


// packages 
// https://onnxruntime.ai/docs/api/csharp/api/Microsoft.ML.OnnxRuntime.html
// dotnet add package Microsoft.ML.OnnxRuntime --version 1.16.0
// dotnet add package Microsoft.ML.OnnxRuntime.Managed --version 1.16.0
// dotnet add package Microsoft.ML
// https://www.nuget.org/packages/SixLabors.ImageSharp
// dotnet add package SixLabors.ImageSharp --version 3.1.4


// Model fined tuned Vision Transformer https://huggingface.co/google/vit-base-patch16-224


using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats;
using SixLabors.ImageSharp.Formats.Png;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace Microsoft.ML.OnnxRuntime.vit_nsfw
{
    class Program
    {
        private static void Main(string[] args)
        {
            if (args.Length == 0)
            {
                Console.WriteLine("Please enter a numeric argument.");

            }
            string modelFilePath = args[0].ToString();
            string imagefilepath = args[1].ToString();


            using Image<Rgb24> image = Image.Load<Rgb24>(imagefilepath);


            using Stream imageStream = new MemoryStream();
            image.Mutate(x =>
            {
                x.Resize(new ResizeOptions
                {
                    Size = new Size(224, 224),
                    Mode = ResizeMode.Crop
                });
            });
            image.Save(imageStream, PngFormat.Instance);
            //image.Save("D:\\repos2\\c#\\restnet\\output\\dog.png");


            // We use DenseTensor for multi-dimensional access to populate the image data
            // Standardization 
            var mean = new[] { 0.485f, 0.456f, 0.406f };
            var stddev = new[] { 0.229f, 0.224f, 0.225f };
            DenseTensor<float> processedImage = new(new[] { 1, 3, 224, 224 });
            image.ProcessPixelRows(accessor =>
            {
                for (int y = 0; y < accessor.Height; y++)
                {
                    Span<Rgb24> pixelSpan = accessor.GetRowSpan(y);
                    for (int x = 0; x < accessor.Width; x++)
                    {
                        processedImage[0, 0, y, x] = ((pixelSpan[x].R / 255f) - mean[0]) / stddev[0];
                        processedImage[0, 1, y, x] = ((pixelSpan[x].G / 255f) - mean[1]) / stddev[1];
                        processedImage[0, 2, y, x] = ((pixelSpan[x].B / 255f) - mean[2]) / stddev[2];
                    }
                }
            });


            // Pin tensor buffer and create a OrtValue with native tensor that makes use of
            // DenseTensor buffer directly. This avoids extra data copy within OnnxRuntime.
            // It will be unpinned on ortValue disposal
            using var inputOrtValue = OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance,
                processedImage.Buffer, new long[] { 1, 3, 224, 224 });

            var inputs = new Dictionary<string, OrtValue>
            {
                { "input", inputOrtValue }
            };


            using var session = new InferenceSession(modelFilePath);
            using var runOptions = new RunOptions();
            using IDisposableReadOnlyCollection<OrtValue> results = session.Run(runOptions, inputs, session.OutputNames);


            // We copy results to array only to apply algorithms, otherwise data can be accessed directly
            // from the native buffer via ReadOnlySpan<T> or Span<T>
            var output = results[0].GetTensorDataAsSpan<float>().ToArray();
            float sum = output.Sum(x => (float)Math.Exp(x));
            IEnumerable<float> softmax = output.Select(x => (float)Math.Exp(x) / sum);

            // Extract top 5 predicted classes
            // IEnumerable Exposes the enumerator, which supports a simple iteration over a collection of a specified type.
            IEnumerable<Prediction> top5 = softmax.Select((x, i) => new Prediction { Label = LabelMap.Labels[i], Confidence = x })
                               .OrderByDescending(x => x.Confidence)
                               .Take(5);

            // Print results to console
            Console.WriteLine("Top 5 predictions for Video Transformer...");
            Console.WriteLine("--------------------------------------------------------------");
            foreach (var t in top5)
            {
                Console.WriteLine($"Label: {t.Label}, Confidence: {t.Confidence}");
            }





        }
    }

}