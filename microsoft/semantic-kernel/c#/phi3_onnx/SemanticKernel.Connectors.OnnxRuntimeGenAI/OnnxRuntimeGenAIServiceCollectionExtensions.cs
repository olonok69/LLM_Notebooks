

using System;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Microsoft.SemanticKernel.ChatCompletion;
using feiyun0112.SemanticKernel.Connectors.OnnxRuntimeGenAI;
using Microsoft.SemanticKernel.Embeddings;
using Microsoft.SemanticKernel.Http;
using Microsoft.SemanticKernel.TextGeneration;

namespace Microsoft.SemanticKernel;

/// <summary>
/// Extension methods for adding OnnxRuntimeGenAI Text Generation service to the kernel builder.
/// </summary>
public static class OnnxRuntimeGenAIServiceCollectionExtensions
{ 
    /// <summary>
    /// Add OnnxRuntimeGenAI Chat Completion services to the specified service collection.
    /// </summary>
    /// <param name="services">The service collection to add the OnnxRuntimeGenAI Text Generation service to.</param>
    /// <param name="modelPath">The generative AI ONNX model path.</param>
    /// <param name="serviceId">Optional service ID.</param>
    /// <returns>The updated service collection.</returns>
    public static IServiceCollection AddOnnxRuntimeGenAIChatCompletion(
        this IServiceCollection services,
        string modelPath,
        string? serviceId = null)
    {
        services.AddKeyedSingleton<IChatCompletionService>(serviceId, (serviceProvider, _) =>
            new OnnxRuntimeGenAIChatCompletionService(
                modelPath,
                loggerFactory: serviceProvider.GetService<ILoggerFactory>()));

        return services;
    }
}
