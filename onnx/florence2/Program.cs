// Florence 2

// https://huggingface.co/microsoft/Florence-2-large
/*Florence-2 is an advanced vision foundation model that uses a prompt-based approach to handle a wide range of vision and vision-language tasks. 
 * Florence-2 can interpret simple text prompts to perform tasks like captioning, object detection, and segmentation. It leverages our FLD-5B dataset, 
 * containing 5.4 billion annotations across 126 million images, to master multi-task learning. The model's sequence-to-sequence architecture enables it to excel in 
 * both zero-shot and fine-tuned settings, proving to be a competitive vision foundation model.
 * 
 * https://arxiv.org/pdf/2311.06242
 * 
 * public enum TaskTypes
{
    OCR,
    OCR_WITH_REGION,
    CAPTION,
    DETAILED_CAPTION,
    MORE_DETAILED_CAPTION,
    OD,
    DENSE_REGION_CAPTION,
    CAPTION_TO_PHRASE_GROUNDING,
    REFERRING_EXPRESSION_SEGMENTATION, 
    REGION_TO_SEGMENTATION,
    OPEN_VOCABULARY_DETECTION,
    REGION_TO_CATEGORY,
    REGION_TO_DESCRIPTION,
    REGION_TO_OCR,
    REGION_PROPOSAL
} 
 * https://github.com/curiosity-ai/florence2-sharp/tree/main
 * Previous video in Python
 * https://github.com/olonok69/LLM_Notebooks/blob/main/microsoft/Florence_Large_Exploration.ipynb
 *
 */


/* dotnet add package Microsoft.ML --version 3.0.1
 * dotnet add package Microsoft.Extensions.Logging --version 8.0.0
 * dotnet add package Microsoft.ML.OnnxRuntime --version 1.18.1
 * dotnet add package Microsoft.ML.OnnxRuntime.Managed --version 1.18.1
 * dotnet add package SixLabors.ImageSharp --version 3.1.5
 * dotnet add package SixLabors.Fonts --version 2.0.4
 * dotnet add package SixLabors.ImageSharp.Drawing --version 2.1.4
 * dotnet add package Zlogger --version 1.7.1
 * dotnet add package Florence2 --version 24.7.50588
 * dotnet add package Spectre.Console --version 0.49.1
 */

using System.Diagnostics;
using System.Numerics;
using System.Text;
using System.Text.Json;
using Florence2;
using Microsoft.Extensions.Logging;
using SixLabors.Fonts;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SixLabors.ImageSharp.Drawing.Processing;
using ZLogger;
using System.Threading.Tasks;
using System.Security.Cryptography.X509Certificates;
using florence2.ConsoleDemo.Utils;
using Spectre.Console;
using System;

public static class Programm
{
    static async Task Main(string[] args)
    {
        ConsoleHelper.ShowHeader();
        using ILoggerFactory factory = LoggerFactory.Create(builder => builder.AddZLoggerConsole());
        ILogger logger = factory.CreateLogger("Florence-2");
        // Get Models Path
        string modelPath  = ConsoleHelper.GetModelsPath(Statics.ModelInputPrompt);
        var modelSource = new FlorenceModelDownloader(modelPath);
        ConsoleHelper.WriteToConsole(Environment.NewLine);
        while (true)

        {
            // Image imput
            string imageInput = ConsoleHelper.GetImagePath(Statics.ImageInput);
            ConsoleHelper.WriteToConsole(Environment.NewLine);

            // Image imput
            string outPath = ConsoleHelper.GetOutputPath(Statics.ImageOutputPath);
            ConsoleHelper.WriteToConsole(Environment.NewLine);
            // Decide if to delete or not
            //if (Directory.Exists(outPath)) Directory.Delete(outPath, true);

            Directory.CreateDirectory(outPath);
            // Image imput
            string task_c = ConsoleHelper.Getprompt(Statics.TaskFlorence);
            ConsoleHelper.WriteToConsole(Environment.NewLine);

            await modelSource.DownloadModelsAsync(status => logger?.ZLogInformation($"{status.Progress:P0} {status.Error} {status.Message}"), logger, CancellationToken.None);
            var model = new Florence2Model(modelSource);

            /*foreach (var task in Enum.GetValues<TaskTypes>())
            {
                using var imgStream = LoadImage("D:\\repos2\\c#\\florence_2\\florence_2\\book.jpg");
                using var imgStreamResult = Image.Load("D:\\repos2\\c#\\florence_2\\florence_2\\book.jpg");

                var results = model.Run(task, imgStream, textInput: "DUANE", CancellationToken.None);
                logger?.ZLogInformation($"{task} : {JsonSerializer.Serialize(results)}");
                Console.WriteLine($"{task} : {results}");
            }*/

            // MORE_DETAILED_CAPTION

            using var imgStream = LoadImage(imageInput);
            using var imgStreamResult = LoadImage(imageInput);
            //TaskTypes task = TaskTypes.MORE_DETAILED_CAPTION;
            TaskTypes task = (TaskTypes)Enum.Parse(typeof(TaskTypes), task_c, true);
            var results = model.Run(task, imgStream, textInput: "DUANE", CancellationToken.None);
            DrawInline(imgStreamResult, task, "window", results, outFolder: outPath);
            //logger?.ZLogInformation($"{task} : {JsonSerializer.Serialize(results)}");
            var output = JsonSerializer.Serialize(results);
            ConsoleHelper.WriteToConsole(Environment.NewLine);
            string OutputToConsole = $"{task} : {output.ToString()}";
            Console.WriteLine(OutputToConsole);
            ConsoleHelper.WriteToConsole(Statics.RestartPrompt);
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
    private static Stream LoadImage(string path) => File.OpenRead(path);
    private static void DrawInline(Stream imgStreamResult, TaskTypes task, string userText, FlorenceResults[] results, string? outFolder = null)
    {
        if (!results.Any(r => (r.OCRBBox is object && r.OCRBBox.Any())
         || (r.BoundingBoxes is object && r.BoundingBoxes.Any())
         || (r.Polygons is object && r.Polygons.Any()))) return;

        outFolder ??= Environment.GetFolderPath(Environment.SpecialFolder.Desktop);

        var penBox = Pens.Solid(SixLabors.ImageSharp.Color.Red, 3.0f);

        if (Florence2Model.TaskPromptsWithoutInputsDict.ContainsKey(task))
        {
            userText = "";
        }

        var fontFamily = DefaultFont.Value;
        var font = fontFamily.CreateFont(50, FontStyle.Italic);

        using (var image = Image.Load<Rgba32>(imgStreamResult))
        {
            image.Mutate(x =>
            {
                for (var index = 0; index < results.Length; index++)
                {
                    var finalResult = results[index];

                    if (finalResult.BoundingBoxes is object)
                    {
                        var i = 0;

                        foreach (var bbox1 in finalResult.BoundingBoxes)
                        {
                            PointF? labelPoint = null;

                            foreach (var bboxBBox in bbox1.BBoxes)
                            {
                                var polygon = new List<PointF>();
                                var p = new PointF(bboxBBox.xmin, bboxBBox.ymin);

                                labelPoint ??= p;

                                polygon.Add(p);
                                polygon.Add(new PointF(bboxBBox.xmin, bboxBBox.ymax));
                                polygon.Add(new PointF(bboxBBox.xmax, bboxBBox.ymax));
                                polygon.Add(new PointF(bboxBBox.xmax, bboxBBox.ymin));

                                x.DrawPolygon(penBox, polygon.ToArray());

                            }

                            var label = bbox1.Label;
                            x.DrawText(label, font, Brushes.Solid(SixLabors.ImageSharp.Color.Black), Pens.Solid(SixLabors.ImageSharp.Color.White, 1), labelPoint.Value);
                            i++;
                        }

                    }

                    if (finalResult.OCRBBox is object)
                    {
                        foreach (var labledOcr in finalResult.OCRBBox)
                        {
                            var polygon = labledOcr.QuadBox.Select(e => new PointF(e.x, e.y)).ToArray();
                            x.DrawPolygon(penBox, polygon);
                            var textZero = polygon.First();
                            x.DrawText(labledOcr.Text, font, Brushes.Solid(SixLabors.ImageSharp.Color.Black), Pens.Solid(SixLabors.ImageSharp.Color.White, 1), textZero);

                        }
                    }

                    if (finalResult.Polygons is object)
                    {
                        foreach (var finalResultPolygon in finalResult.Polygons)
                        {
                            PointF? labelPoint = null;

                            if (finalResultPolygon.Polygon is object)
                            {
                                var polygon1 = finalResultPolygon.Polygon.Select(e =>
                                {
                                    var p = new PointF(e.x, e.y);
                                    labelPoint ??= p;
                                    return p;
                                }).ToArray();
                                x.DrawPolygon(penBox, polygon1);
                            }

                            if (finalResultPolygon.BBoxes is object)
                            {
                                foreach (var bboxBBox in finalResultPolygon.BBoxes)
                                {
                                    var polygon = new List<PointF>();
                                    var p = new PointF(bboxBBox.xmin, bboxBBox.ymin);

                                    labelPoint ??= p;

                                    polygon.Add(p);
                                    polygon.Add(new PointF(bboxBBox.xmin, bboxBBox.ymax));
                                    polygon.Add(new PointF(bboxBBox.xmax, bboxBBox.ymax));
                                    polygon.Add(new PointF(bboxBBox.xmax, bboxBBox.ymin));

                                    x.DrawPolygon(penBox, polygon.ToArray());

                                }
                            }

                            x.DrawText(finalResultPolygon.Label, font, Brushes.Solid(SixLabors.ImageSharp.Color.Black), Pens.Solid(SixLabors.ImageSharp.Color.White, 1), labelPoint.Value);
                        }
                    }

                }
            });

            image.SaveAsBmp($"{outFolder}/book-{task}-{userText}.bmp");
        }
    }

    private static Lazy<FontFamily> DefaultFont = new Lazy<FontFamily>(() => GetDefaultFont());
    private static FontFamily GetDefaultFont()
    {
        FontFamily? best = null;

        if (OperatingSystem.IsWindows() || OperatingSystem.IsMacOS())
        {
            best = SystemFonts.Get("Arial");
        }
        else if (OperatingSystem.IsLinux())
        {
            best = SystemFonts.TryGet("Arial", out var arial) ? arial :
                SystemFonts.TryGet("Ubuntu", out var sf) ? sf :
                SystemFonts.TryGet("Liberation Sans", out var ls) ? ls :
                SystemFonts.TryGet("DejaVu Sans", out var dvs) ? dvs :
                SystemFonts.TryGet("Rasa", out var rasa) ? rasa :
                SystemFonts.TryGet("FreeSans", out var fs) ? fs :
                                                                    null;
        }
        return best ?? SystemFonts.Families.FirstOrDefault(f => f.Name.Contains("Sans"), SystemFonts.Families.First());
    }



}