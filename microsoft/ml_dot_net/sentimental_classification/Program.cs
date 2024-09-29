/*
 * ML.NET is an open-source, cross-platform machine learning framework for .NET developers that enables integration of custom machine learning models into .NET applications. 
 * It encompasses an API, which consists of different NuGet packages, a Visual Studio extension called Model Builder, and a command-line interface that's installed as a .NET tool.
 * 
 * Features
 * ========
 * Custom ML made easy with AutoML
   ML.NET offers Model Builder (a simple UI tool) and ML.NET CLI to make it super easy to build custom ML Models.
   These tools use Automated ML (AutoML), a cutting edge technology that automates the process of building best performing models for your Machine Learning scenario. 
   All you have to do is load your data, and AutoML takes care of the rest of the model building process.

 * Extended with TensorFlow & more
   ML.NET has been designed as an extensible platform so that you can consume other popular ML frameworks (TensorFlow, ONNX, Infer.NET, and more) and have access 
   to even more machine learning scenarios, like image classification, object detection, and more.

 * High performance and accuracy

 * */

// dotnet add package Microsoft.ML.TorchSharp --version 0.22.0-preview.24378.1
using System;
using System.IO;
using Microsoft.ML;
using SentimentAnalysisConsoleApp.DataStructures;
using Common;
using static Microsoft.ML.DataOperationsCatalog;

namespace SentimentAnalysisConsoleApp
{
    internal static class Program
    {
        // Data path
        private static readonly string DataRelativePath = @"D:\repos2\c#\Data\wikiDetoxAnnotated40kRows.tsv";
        private static readonly string DataPath = GetAbsolutePath(DataRelativePath);
        //Model Output path
        private static readonly string ModelRelativePath = @"D:\repos2\c#\MLModels\SentimentModel.zip";
        private static readonly string ModelPath = GetAbsolutePath(ModelRelativePath);

        static void Main(string[] args)
        {
            // Create MLContext to be shared across the model creation workflow objects 
            // Set a random seed for repeatable/deterministic results across multiple trainings.
            var mlContext = new MLContext(seed: 1);

            // STEP 1: Common data loading configuration
            IDataView dataView = mlContext.Data.LoadFromTextFile<SentimentIssue>(DataPath, hasHeader: true);

            TrainTestData trainTestSplit = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
            IDataView trainingData = trainTestSplit.TrainSet;
            IDataView testData = trainTestSplit.TestSet;

            // STEP 2: Common data process configuration with pipeline data transformations          
            var dataProcessPipeline = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(SentimentIssue.Text));

            // STEP 3: Set the training algorithm, then create and config the modelBuilder                            
            var trainer = mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features");
            var trainingPipeline = dataProcessPipeline.Append(trainer);

            // STEP 4: Train the model fitting to the DataSet
            ITransformer trainedModel = trainingPipeline.Fit(trainingData);

            // STEP 5: Evaluate the model and show accuracy stats
            var predictions = trainedModel.Transform(testData);
            var metrics = mlContext.BinaryClassification.Evaluate(data: predictions, labelColumnName: "Label", scoreColumnName: "Score");

            ConsoleHelper.PrintBinaryClassificationMetrics(trainer.ToString(), metrics);

            // STEP 6: Save/persist the trained model to a .ZIP file
            mlContext.Model.Save(trainedModel, trainingData.Schema, ModelPath);

            Console.WriteLine("The model is saved to {0}", ModelPath);

            // Make a single test prediction, loading the model from .ZIP file
            SentimentIssue sampleStatement = new SentimentIssue { Text = "I love this movie!" };

            // Create prediction engine related to the loaded trained model
            var predEngine = mlContext.Model.CreatePredictionEngine<SentimentIssue, SentimentPrediction>(trainedModel);

            // Score
            var resultprediction = predEngine.Predict(sampleStatement);

            Console.WriteLine($"=============== Single Prediction  ===============");
            Console.WriteLine($"Text: {sampleStatement.Text} | Prediction: {(Convert.ToBoolean(resultprediction.Prediction) ? "Toxic" : "Non Toxic")} sentiment | Probability of being toxic: {resultprediction.Probability} ");
            Console.WriteLine($"================End of Process.Hit any key to exit==================================");
            Console.ReadLine();
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
