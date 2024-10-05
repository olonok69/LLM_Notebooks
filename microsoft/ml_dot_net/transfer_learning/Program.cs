// dotnet add package Microsoft.ML --version 4.0.0-preview.24378.1
// dotnet add package Microsoft.ML.ImageAnalytics --version 4.0.0-preview.24378.1
//dotnet add package Microsoft.ML.Vision --version 4.0.0-preview.24378.1
// dotnet add package SciSharp.TensorFlow.Redist-Windows-GPU --version 2.10.3
//dotnet add package Microsoft.ML.TensorFlow --version 4.0.0-preview.24378.1
// dotnet add package Spectre.Console --version 0.49.1
//
// Dataset https://digitalcommons.usu.edu/all_datasets/48/ SDNET2018: A concrete crack image dataset for machine learning applications
/* Citation 
S. Dorafshan and M. Maguire, "Autonomous detection of concrete cracks on bridge decks and fatigue cracks on steel members," in Digital Imaging 2017, Mashantucket, CT, 2017. 
S. Dorafshan, M. Maguire and M. Chang, "Comparing automated image-based crack detection techniques in spatial and frequency domains," in Proceedings of the 26th American Society of Nondestructive Testing Research Symposium, Jacksonville, FL, 2017. 
S. Dorafshan, M. Maguire, N. Hoffer and C. Coopmans, "Challenges in bridge inspection using small unmanned aerial systems: Results and lessons learned," in Proceedings of the 2017 International Conference on Unmanned Aircraft Systems, Miami, FL, 2017. 
S. Dorafshan, C. Coopmans, R. J. Thomas and M. Maguire, "Deep Learning Neural Networks for sUAS-Assisted Structural Inspections, Feasibility and Application," in ICUAS 2018, Dallas, TX, 2018. 
S. Dorafshan, M. Maguire and X. Qi, "Automatic Surface Crack Detection in Concrete Structures Using OTSU Thresholding and Morphological Operations," Utah State University, Logan, Utah, USA, 2016.
S. Dorafshan, J. R. Thomas and M. Maguire, "Comparison of Deep Learning Convolutional Neural Networks and Edge Detectors for Image-Based Crack Detection in Concrete," Submitted to Journal of Construction and Building Materials, 2018. 
S. Dorafshan, R. Thomas and M. Maguire, "Image Processing Algorithms for Vision-based Crack Detection in Concrete Structures," Submitted to Advanced Concrete Technology, 2018.  
*/
using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using Microsoft.ML;
using static Microsoft.ML.DataOperationsCatalog;
using Microsoft.ML.Vision;
using System.Data;
using mlnet_image_classification.ConsoleDemo.Utils;
using Spectre.Console;
using static System.Runtime.InteropServices.JavaScript.JSType;
using Tensorflow;

namespace DeepLearning_ImageClassification
{
    class Program
    {
        static void Main(string[] args)
        {
            ConsoleHelper.ShowHeader();
            ConsoleHelper.WriteToConsole(Environment.NewLine);

            var workspaceRelativePath = @"D:\repos2\c#\mlnet_image_classification\mlnet_image_classification\model\";
            var assetsRelativePath = @"D:\repos2\c#\mlnet_image_classification\mlnet_image_classification\assets\";
            var model_path = Path.Combine(workspaceRelativePath, "model.zip");
            MLContext mlContext = new MLContext();
            ConsoleHelper.WriteToConsole("Created Context Dataset");
            // You must unzip assets.zip before training
            IEnumerable<ImageData> images = LoadImagesFromDirectory(folder: assetsRelativePath, useFolderNameAsLabel: true);

            IDataView imageData = mlContext.Data.LoadFromEnumerable(images);
            IDataView shuffledData = mlContext.Data.ShuffleRows(imageData);

            var preprocessingPipeline = mlContext.Transforms.Conversion.MapValueToKey(
                    inputColumnName: "Label",
                    outputColumnName: "LabelAsKey")
                .Append(mlContext.Transforms.LoadRawImageBytes(
                    outputColumnName: "Image",
                    imageFolder: assetsRelativePath,
                    inputColumnName: "ImagePath"));

            IDataView preProcessedData = preprocessingPipeline
                                .Fit(shuffledData)
                                .Transform(shuffledData);

            TrainTestData trainSplit = mlContext.Data.TrainTestSplit(data: preProcessedData, testFraction: 0.3);
            TrainTestData validationTestSplit = mlContext.Data.TrainTestSplit(trainSplit.TestSet);

            IDataView trainSet = trainSplit.TrainSet;
            IDataView validationSet = validationTestSplit.TrainSet;
            IDataView testSet = validationTestSplit.TestSet;
            ConsoleHelper.WriteToConsole("Loaded Dataset");


            string task = ConsoleHelper.GetTask(Statics.TaskClassifier);
            if (task == "1")
            {
                //ImageClassificationTrainer.Architecture.
                var classifierOptions = new ImageClassificationTrainer.Options()
                {
                    FeatureColumnName = "Image",
                    LabelColumnName = "LabelAsKey",
                    ValidationSet = validationSet,
                    Arch = ImageClassificationTrainer.Architecture.ResnetV2101,
                    MetricsCallback = (metrics) => Console.WriteLine(metrics),
                    TestOnTrainSet = false,
                    ReuseTrainSetBottleneckCachedValues = true,
                    ReuseValidationSetBottleneckCachedValues = true
                };

                var trainingPipeline = mlContext.MulticlassClassification.Trainers.ImageClassification(classifierOptions)
                    .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
                ITransformer trainedModel = trainingPipeline.Fit(trainSet);

                mlContext.Model.Save(trainedModel, trainSplit.TrainSet.Schema, model_path);
                var message = $"Model saved to {model_path}";
                ConsoleHelper.WriteToConsole(message);

                Console.ReadKey();
            }
            else if (task == "2")
            {
                // inference only 1 image
                DataViewSchema modelSchema;
                ITransformer loadedModel = mlContext.Model.Load(model_path, out modelSchema);
                ConsoleHelper.ShowHeader();
                ConsoleHelper.WriteToConsole(Environment.NewLine);
                ConsoleHelper.WriteToConsole($"Model Loaded from: {model_path}");
                var key = Console.ReadKey();
                string taskinference = ConsoleHelper.GetTask(Statics.TaskInference);
                if (taskinference == "1")
                {
                    ConsoleHelper.WriteToConsole(Environment.NewLine);

                    ClassifyImages(mlContext, testSet, loadedModel);
                    Console.ReadKey();
                }
                else if (taskinference == "2")
                {
                    PredictionEngine<ModelInput, ModelOutput> predictionEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(loadedModel);
                    ModelInput image = mlContext.Data.CreateEnumerable<ModelInput>(testSet, reuseRowObject: true).First();
                    ModelOutput prediction = predictionEngine.Predict(image);

                    ConsoleHelper.WriteToConsole(Environment.NewLine);
                    ConsoleHelper.WriteToConsole("Classifying single Image");
                    OutputPrediction(prediction);
                    ConsoleHelper.WriteToConsole(Environment.NewLine);
                    Console.ReadKey();
                }
            }
            else
            {
                Console.ReadKey();
            }


        }

        public static void ClassifySingleImage(MLContext mlContext, IDataView data, ITransformer trainedModel)
        {
            PredictionEngine<ModelInput, ModelOutput> predictionEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(trainedModel);

            ModelInput image = mlContext.Data.CreateEnumerable<ModelInput>(data, reuseRowObject: true).First();

            ModelOutput prediction = predictionEngine.Predict(image);

            Console.WriteLine("Classifying single image");
            OutputPrediction(prediction);
        }

        public static void ClassifyImages(MLContext mlContext, IDataView data, ITransformer trainedModel)
        {
            IDataView predictionData = trainedModel.Transform(data);

            IEnumerable<ModelOutput> predictions = mlContext.Data.CreateEnumerable<ModelOutput>(predictionData, reuseRowObject: true).Take(10);

            Console.WriteLine("Classifying multiple images");
            foreach (var prediction in predictions)
            {
                OutputPrediction(prediction);
            }
        }

        private static void OutputPrediction(ModelOutput prediction)
        {
            string imageName = Path.GetFileName(prediction.ImagePath);
            Console.WriteLine($"Image: {imageName} | Actual Value: {prediction.Label} | Predicted Value: {prediction.PredictedLabel}");
        }

        public static IEnumerable<ImageData> LoadImagesFromDirectory(string folder, bool useFolderNameAsLabel = true)
        {
            var files = Directory.GetFiles(folder, "*",
                searchOption: SearchOption.AllDirectories);

            foreach (var file in files)
            {
                if ((Path.GetExtension(file) != ".jpg") && (Path.GetExtension(file) != ".png"))
                    continue;

                var label = Path.GetFileName(file);

                if (useFolderNameAsLabel)
                    label = Directory.GetParent(file).Name;
                else
                {
                    for (int index = 0; index < label.Length; index++)
                    {
                        if (!char.IsLetter(label[index]))
                        {
                            label = label.Substring(0, index);
                            break;
                        }
                    }
                }

                yield return new ImageData()
                {
                    ImagePath = file,
                    Label = label
                };
            }
        }
    }

    class ImageData
    {
        public string ImagePath { get; set; }

        public string Label { get; set; }
    }

    class ModelInput
    {
        public byte[] Image { get; set; }

        public UInt32 LabelAsKey { get; set; }

        public string ImagePath { get; set; }

        public string Label { get; set; }
    }

    class ModelOutput
    {
        public string ImagePath { get; set; }

        public string Label { get; set; }

        public string PredictedLabel { get; set; }
    }
}
