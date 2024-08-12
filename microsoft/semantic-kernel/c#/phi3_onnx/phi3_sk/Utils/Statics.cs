using System.Threading.Tasks;

namespace Phi3OnnxConsole.Utils;

internal class Statics
{
    /// <summary>
    ///     The system prompt.
    /// </summary>
    public const string SystemPrompt
        = "You are an AI assistant that helps people find information. " +
        "Answer the questions briefly.";

    /// <summary>
    ///     Start prompt.
    /// </summary>
    public const string StartChat
        = "@Ask your question.Type an empty string to Exit.";

    /// <summary>
    ///     Query User prompt.
    /// </summary>
    public const string QueryChat
        = "@Q: ";
    /// <summary>
    ///     Response Phi User prompt.
    /// </summary>
    public const string PhiChat
        = "@Phi3: ";


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
        "Press any key to send another questionr or ESC to end.";

    /// <summary>
    ///     The User Prompt indicating 
    /// </summary>
    public const string UserPrompt =
        "@User: ";
    /// <summary>
    ///     The User Prompt indicating 
    /// </summary>
    public const string Execute_Task =
        "Working in your Request.......";
}