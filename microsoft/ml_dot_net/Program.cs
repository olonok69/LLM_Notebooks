// dotnet add package Microsoft.ML --version 3.0.1
// dotnet add package Microsoft.ML.TensorFlow --version 3.0.1
// dotnet add package Microsoft.ML.ImageAnalytics --version 3.0.1
// dotnet add package SciSharp.TensorFlow.Redist --version 2.16.0

/* Links
 http://tersorflow.org
 TensorFlow is an open-source machine learning library developed by Google. TensorFlow is used to build and train deep learning models as it facilitates the creation of 
 computational graphs and efficient execution on various hardware platforms. The article provides an comprehensive overview of tensorflow.
 *https://github.com/SciSharp/TensorFlow.NET  
 TensorFlow.NET (TF.NET) provides a .NET Standard binding for TensorFlow. It aims to implement the complete Tensorflow API in C# which allows .NET developers 
 to develop, train and deploy Machine Learning models with the cross-platform .NET Standard framework. TensorFlow.NET has built-in Keras high-level interface and 
 is released as an independent package TensorFlow.Keras.
 
 ML.Net https://dotnet.microsoft.com/en-us/apps/machinelearning-ai/ml-dotnet
 Microsoft.ML.TensorFlow  contains ML.NET integration of TensorFlow
 Microsoft.ML.ImageAnalytics work with Images
 */

using ImageClassification.ModelScorer;
using System;
using System.IO;


namespace ImageClassification
{
    public class Program
    {
        static void Main(string[] args)
        {
            string assetsRelativePath = @"D:\repos2\c#\ImageClassification\assets\";
            string assetsPath = GetAbsolutePath(assetsRelativePath);

            var tagsTsv = Path.Combine(assetsPath, "inputs", "images", "tags.tsv");
            var imagesFolder = Path.Combine(assetsPath, "inputs", "images");
            var inceptionPb = Path.Combine(assetsPath, "inputs", "inception", "tensorflow_inception_graph.pb");
            var labelsTxt = Path.Combine(assetsPath, "inputs", "inception", "imagenet_comp_graph_label_strings.txt");

            try
            {
                var modelScorer = new TFModelScorer(tagsTsv, imagesFolder, inceptionPb, labelsTxt);
                modelScorer.Score();

            }
            catch (Exception ex)
            {
                ConsoleHelpers.ConsoleWriteException(ex.ToString());
            }

            ConsoleHelpers.ConsolePressAnyKey();
        }

        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;
            string fullPath = Path.Combine(assemblyFolderPath, relativePath);
            return fullPath;
        }
    }
}
