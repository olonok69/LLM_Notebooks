using System.Threading.Tasks;

namespace mlnet_image_classification.ConsoleDemo.Utils;

internal class Statics
{
    
    /// <summary>
    ///     The prompt indicating the output section.
    /// </summary>
    public const string RestartPrompt =
        "Press any key to analize another image or ESC to end.";


    /// <summary>
    ///     Ask Model Language to use in OCR.
    /// </summary>
    public const string TaskClassifier
        = "Enter [green]1--> Train Model, 2--> Inference,  Esc--> Exit [/]:";

    public const string TaskInference
     = "Enter [green]1--> Inference  multiple Image, 2--> Inference 1 Image,  Esc--> Exit [/]:";


}