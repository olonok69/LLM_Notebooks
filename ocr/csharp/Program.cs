/* Tesseract was originally developed at Hewlett-Packard Laboratories Bristol UK and at Hewlett-Packard Co, Greeley Colorado USA between 1985 and 1994, 
 * with some more changes made in 1996 to port to Windows, and some C++izing in 1998. In 2005 Tesseract was open sourced by HP. From 2006 until November 2018 
 * it was developed by Google.
 https://github.com/tesseract-ocr/tesseract
 */

// Packages dotnet add package TesseractOCR --version 5.3.5
// dotnet add package Spectre.Console --version 0.49.1
/* The DLL's Tesseract53.dll (and exe) and leptonica-1.83.0.dll are compiled with Visual Studio 2022 you need these C++ runtimes for it on your computer

X86: https://aka.ms/vs/17/release/vc_redist.x86.exe
X64: https://aka.ms/vs/17/release/vc_redist.x64.exe
*/

// MODELS https://github.com/tesseract-ocr/tessdata


// DOC https://tesseract-ocr.github.io/tessdoc/Command-Line-Usage.html

using System;
using System.Diagnostics;
using TesseractOCR.Enums;
using TesseractOCR;
using Spectre.Console;
using Tesseract.ConsoleDemo.Utils;
using System.IO;
using System.Text;


ConsoleHelper.ShowHeader();

// Get the model path from the user
string modelPath
    = ConsoleHelper.GetFolderPath(Statics.ModelInputPrompt);
// Show the header
ConsoleHelper.ShowHeader();
while (true)
{
    // Get Image Path
    ConsoleHelper.WriteToConsole(Environment.NewLine);
    var testImagePath = ConsoleHelper.Getprompt(Statics.ImageInput);
    ConsoleHelper.ShowHeader();
    // Get Language
    ConsoleHelper.WriteToConsole(Environment.NewLine);
    var language = ConsoleHelper.GetLanguage(Statics.Language);
    ConsoleHelper.WriteToConsole(Environment.NewLine);

    try
    {
        using (var engine = new Engine(modelPath, language, EngineMode.Default))
        {
            using (var img = TesseractOCR.Pix.Image.LoadFromFile(testImagePath))
            {
                using (var page = engine.Process(img))
                {
                    // print confidence Score
                    string confidence_text = $"Mean confidence: {page.MeanConfidence}";
                    ConsoleHelper.WriteToConsole(confidence_text);
                    // Get user question
                    ConsoleHelper.WriteToConsole(Environment.NewLine);
                    ConsoleHelper.WriteToConsole("Press a ke to Continue");
                    var key2 = Console.ReadKey();
                    // Get Output Path
                    var outputImagePath = ConsoleHelper.Getprompt(Statics.textOutput);
                    string texto = page.Text as string;
                    string dataasstring = $"{texto}";
                    // Write File to disk
                    using (FileStream fs = File.Create(outputImagePath))
                    {
                        // writing data in string

                        byte[] info = new UTF8Encoding(true).GetBytes(dataasstring);
                        fs.Write(info, 0, info.Length);

                        // writing data in bytes already
                        byte[] data = new byte[] { 0x0 };
                        fs.Write(data, 0, data.Length);
                    }
                    ConsoleHelper.WriteToConsole(Environment.NewLine);
                }
            }
        }
    }
    catch (Exception ex)
    {
        ConsoleHelper.WriteToConsole(Environment.NewLine);
        string exception = $"Exception:\r\n{ex}";
        ConsoleHelper.WriteToConsole(exception);
    }
    ConsoleHelper.WriteToConsole(Environment.NewLine);
    ConsoleHelper.WriteToConsole(Statics.RestartPrompt);
    ConsoleHelper.WriteToConsole(Environment.NewLine);
    var key = Console.ReadKey();
    if (key.Key == ConsoleKey.Escape)
    {
        break;
    }
}