using Spectre.Console;

namespace Phi3VisionOnnxConsole.Utils;

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
        grid.AddRow(new FigletText("Phi-3 Vision ONNX").Centered().Color(Color.NavajoWhite3));
        grid.AddRow(new Text("Steps:", new Style(Color.BlueViolet, Color.Black)).LeftJustified());
        grid.AddRow(new Text("1.- Introduce path Model. You enter in a loop",  new Style(Color.Red, Color.Black)).LeftJustified());
        grid.AddRow(new Text("2.- Introduce path Image", new Style(Color.Red, Color.Black)).LeftJustified());
        grid.AddRow(new Text("3.- Introduce prompt for this Image", new Style(Color.Red, Color.Black)).LeftJustified());
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
    public static string GetFilePath(string prompt)
    {
        ShowHeader();

        return AnsiConsole.Prompt(
            new TextPrompt<string>(prompt)
            .PromptStyle("white")
            .ValidationErrorMessage("[red]Invalid path[/]")
            .Validate(filePath =>
            {
                if (!File.Exists(filePath))
                {
                    return ValidationResult.Error("[red]File does not exist[/]");
                }

                if (!filePath.EndsWith(".jpg", StringComparison.OrdinalIgnoreCase) &&
                    !filePath.EndsWith(".jpeg", StringComparison.OrdinalIgnoreCase) &&
                    !filePath.EndsWith(".png", StringComparison.OrdinalIgnoreCase))
                {
                    return ValidationResult.Error("[red]File is not a picture[/]");
                }

                return ValidationResult.Success();
            }));
    }
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