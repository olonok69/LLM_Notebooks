// dotnet add package Panlingo.LanguageIdentification.CLD3 --version 0.0.0.18
// https://github.com/gluschenko/language-identification/tree/master

/* Install
 * 
 * sudo apt -y install protobuf-compiler libprotobuf-dev nuget
 * 
 * wget https://packages.microsoft.com/config/debian/12/packages-microsoft-prod.deb -O packages-microsoft-prod.deb
 * sudo dpkg -i packages-microsoft-prod.deb
 * rm packages-microsoft-prod.de
 * sudo apt-get update &&   sudo apt-get install -y dotnet-sdk-8.0
 * 
 * Models
 * curl --location -o /models/fasttext176.bin https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
 * curl --location -o /models/fasttext217.bin https://huggingface.co/facebook/fasttext-language-identification/resolve/main/model.bin?download=true
 * 
 * build app in LInux
 * dotnet new create console -n lang_detector
 *   cd lang_detector
 *  dotnet add package Panlingo.LanguageIdentification.FastText
 *  dotnet add package  Mosaik.Core --version 24.8.51117
 *  dotnet run
 *  dotnet lang_detector.dll
 *  
 */

using Panlingo.LanguageIdentification.FastText;
namespace languages.LanguageDetection;
class Program
{
    static void Main()
    {
        using var fastText = new FastTextDetector();
        fastText.LoadModel("/home/olonok/lang_detector/fasttext/lang_detector/models/fasttext217.bin");
        foreach (var (lang, texto) in Data.ShortSamples)
        {
            var predictions = fastText.Predict(
            text: texto,
            count: 1
        );

            foreach (var prediction in predictions)
            {
                Console.WriteLine($"Original Language Label: {lang}, Predicted Label: {prediction.Label}: Predicted Probability: {prediction.Probability}");
            }

            var dimensions = fastText.GetModelDimensions();
            var labels = fastText.GetLabels();
        }
    }
}
