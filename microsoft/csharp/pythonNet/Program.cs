/* https://spacy.io/usage#quickstart
 * 
 * packages
 * dotnet add package pythonnet --version 3.1.0-preview2024-09-06
 * https://github.com/pythonnet/pythonnet
 * https://pythonnet.github.io/pythonnet/dotnet.html
 * Set PYTHONHOME PYTHONPATH and PYTHONNET_PYDLL
 */

using System;
using System.Threading.Tasks;
using Python.Runtime;


namespace Catalyst.Spacy_Test
{
    class Program
    {
        public static async Task Main(string[] args)

        {
            //initialize python engone and print system information
            Runtime.PythonDLL = @"C:\Users\User\.conda\envs\dotnet\python38.dll";
            var pathToVirtualEnv = @"C:\Users\User\.conda\envs\dotnet";
            Console.WriteLine("\n Loading pythonNet and testing\n");
            Console.WriteLine(pathToVirtualEnv);
            Console.WriteLine(Runtime.PythonDLL);
            Console.WriteLine(PythonEngine.Platform);
            Console.WriteLine(PythonEngine.MinSupportedVersion);
            Console.WriteLine(PythonEngine.MaxSupportedVersion);
            Console.WriteLine(PythonEngine.BuildInfo);
            Console.WriteLine(PythonEngine.PythonPath);

            string additional = $"{pathToVirtualEnv};{pathToVirtualEnv}\\Lib\\site-packages;{pathToVirtualEnv}\\Lib";
            PythonEngine.PythonPath = PythonEngine.PythonPath + ";" + additional;
            Console.WriteLine(PythonEngine.PythonPath);

            PythonEngine.Initialize();
            PythonEngine.BeginAllowThreads();
            Console.ReadKey();

            using (Py.GIL())
            {
                dynamic np = Py.Import("numpy");
                dynamic spacy = Py.Import("spacy");
                dynamic pandas = Py.Import("pandas");

                // Spacy
                Console.WriteLine("\n Loading Spacy and testing\n");
                var nlp = spacy.load("en_core_web_sm");
                var doc = nlp("this is a test of spacy in csharp");
                Console.WriteLine(doc.text);
                Console.WriteLine("\n");
                Console.ReadKey();
                //numpy
                Console.WriteLine("\n Loading Numpy and testing\n");
                Console.WriteLine(np.cos(np.pi * 2));
                dynamic sin = np.sin;
                Console.WriteLine(sin(5));
                double c = (double)(np.cos(5) + sin(5));
                Console.WriteLine(c);
                dynamic a = np.array(new List<float> { 1, 2, 3 });
                Console.WriteLine(a.dtype);
                dynamic b = np.array(new List<float> { 6, 5, 4 }, dtype: np.int32);
                Console.WriteLine(b.dtype);
                Console.WriteLine(a * b);
                Console.ReadKey();

                // pandas
                Console.WriteLine("\n Loading Pandas and testing\n");
                var df = pandas.read_csv(@"D:\repos2\c#\spacycsharp\spacy_csharp\APPL.csv");
                Console.WriteLine(df.head());
            }
            


        }
    }
}