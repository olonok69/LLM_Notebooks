namespace Phi3VisionOnnxConsole.Utils;

internal class Statics
{
    /// <summary>
    ///     The system prompt.
    /// </summary>
    public const string SystemPrompt
        = "You are an AI assistant that helps people find information. " +
        "Answer the questions briefly.";

    /// <summary>
    ///     The user prompt.
    /// </summary>
    public const string UserImagePrompt
        = "Describe the image.";

    /// <summary>
    ///     The prompt asking the user to enter the path to the picture file.
    /// </summary>
    public const string PictureInputPrompt
        = "Enter the path to the [yellow]picture file[/]:";

    /// <summary>
    ///     The message displayed when the model is being loaded.
    /// </summary>
    public const string ModelLoadingMessage
        = "[yellow]Loading Model...[/]";

    /// <summary>
    ///     The message displayed when the image is being analyzed.
    /// </summary>
    public const string AnalyzeImageMessage
        = "[yellow]Analyze Image...[/]";

    /// <summary>
    ///     The prompt asking the user to enter the path to the model folder.
    /// </summary>
    public const string ModelInputPrompt
        = "Enter the path to the [yellow]model folder[/]:";

    /// <summary>
    ///     The prompt indicating the output section.
    /// </summary>
    public const string OutputPrompt =
        "[green]Output:[/]";

    /// <summary>
    ///     The prompt indicating the output section.
    /// </summary>
    public const string RestartPrompt =
        "Press any key to analyze another image.";

    /// <summary>
    ///     The User Prompt indicating 
    /// </summary>
    public const string UserPrompt =
        "What do you want to do with this image.?";
    /// <summary>
    ///     The User Prompt indicating 
    /// </summary>
    public const string Execute_Task =
        "Working in your Request.......";
}