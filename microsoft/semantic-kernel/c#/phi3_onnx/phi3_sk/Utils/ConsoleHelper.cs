using Spectre.Console;

namespace Phi3OnnxConsole.Utils;

internal static class ConsoleHelper
{
    /// <summary>
    ///     Clears the console and creates the header for the application.
    /// </summary>
    public static void ShowHeader()
    {
        AnsiConsole.Clear();

        Grid grid = new();
        grid.AddColumn();
        grid.AddRow(new FigletText("Chatting with Phi-3 4k-Instruct ONNX").Centered().Color(Color.Aquamarine1));
        grid.AddRow(new Text("Steps:", new Style(Color.BlueViolet, Color.Black)).LeftJustified());
        grid.AddRow(new Text("1.- Introduce path Model. You enter in a loop",  new Style(Color.Red, Color.Black)).LeftJustified());
        grid.AddRow(new Text("2.- Introduce query after the @User:", new Style(Color.Red, Color.Black)).LeftJustified());
        grid.AddRow(new Text("To exit press Scape", new Style(Color.Red, Color.Black)).LeftJustified());

        AnsiConsole.Write(grid);
        AnsiConsole.WriteLine();
    }

    /// <summary>
    ///     Gets the folder path from the user.
    /// </summary>
    /// <param name="prompt">The prompt message.</param>
    /// <returns>The folder path entered by the user.</returns>
    public static string GetFolderPath(string prompt)
    {
        ShowHeader();

        return AnsiConsole.Prompt(
            new TextPrompt<string>(prompt)
            .PromptStyle("white")
            .ValidationErrorMessage("[red]Invalid path[/]")
            .Validate(dictPath =>
            {
                if (!Directory.Exists(dictPath))
                {
                    return ValidationResult.Error("[red]Path does not exist[/]");
                }

                return ValidationResult.Success();
            }));
    }


    /// <summary>
    ///     Gets the file path from the user.
    /// </summary>
    /// <param name="prompt">The prompt message.</param>
    /// <returns>The file path entered by the user.</returns>

    public static string Getprompt(string prompt)
    {
        ShowHeader();

        return AnsiConsole.Prompt(
            new TextPrompt<string>(prompt)
            .PromptStyle("red")
            );
    }
    /// <summary>
    ///     Writes the specified text to the console.
    /// </summary>
    /// <param name="text">The text to write.</param>
    public static void WriteToConsole(string text)
    {
        AnsiConsole.Markup($"[white]{text}[/]");
    }
}