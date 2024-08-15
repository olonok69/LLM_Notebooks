using Spectre.Console;

namespace Tesseract.ConsoleDemo.Utils;

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
        grid.AddRow(new FigletText("Tessearct OCR").Centered().Color(Color.Aquamarine1));
        grid.AddRow(new Text("Steps:", new Style(Color.BlueViolet, Color.Black)).LeftJustified());
        grid.AddRow(new Text("1.- Introduce path Models",  new Style(Color.Red, Color.Black)).LeftJustified());
        grid.AddRow(new Text("2.- Introduce path Image", new Style(Color.Red, Color.Black)).LeftJustified());
        grid.AddRow(new Text("3.- Introduce Language", new Style(Color.Red, Color.Black)).LeftJustified());
        grid.AddRow(new Text("4.- Introduce path Text File", new Style(Color.Red, Color.Black)).LeftJustified());

        AnsiConsole.Write(grid);
        AnsiConsole.WriteLine();
    }


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


    public static string Getprompt(string prompt)
    {
        ShowHeader();

        return AnsiConsole.Prompt(
            new TextPrompt<string>(prompt)
            .PromptStyle("red")
            );
    }
    public static string GetLanguage(string prompt)
    {
        ShowHeader();

        return AnsiConsole.Prompt(
            new TextPrompt<string>(prompt)
            .PromptStyle("blue")
            );
    }
    public static string GetOutputPath(string prompt)
    {
        ShowHeader();

        return AnsiConsole.Prompt(
            new TextPrompt<string>(prompt)
            .PromptStyle("green")
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