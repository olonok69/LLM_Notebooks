// Create plugins for semantic kernel
//
// built-in plugins , Custom Plugins
// 

// In the Semantic Kernel SDK, a plugin is a class that contains functions that can be run by the kernel.
// A plugin function can be made from a semantic prompt or native function code. To use a plugin, you add it to the kernel and then call the desired function using the
// InvokeAsync method. The kernel accesses the plugin, locates and runs the function, then returns the result. Let's take a look at some predefined plugins.


// Install
//dotnet add package Microsoft.SemanticKernel
//dotnet add package Microsoft.SemanticKernel.Plugins.Core --version 1.2.0-alpha
// https://learn.microsoft.com/en-us/dotnet/api/microsoft.semantickernel.plugins.core?view=semantic-kernel-dotnet

// https://learn.microsoft.com/en-us/semantic-kernel/prompts/
// https://learn.microsoft.com/en-us/semantic-kernel/agents/plugins
// https://learn.microsoft.com/en-us/semantic-kernel/prompts/saving-prompts-as-files

using System;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Plugins.Core;
using Microsoft.SemanticKernel.ChatCompletion;

var builder = Kernel.CreateBuilder();

var azureopenaikey = Environment.GetEnvironmentVariable("AZURE_OPENAI_API_KEY");

builder.AddAzureOpenAIChatCompletion(
         "gpt-3-5-16k",                      // Azure OpenAI Deployment Name
         "https://open-ai-olonok.openai.azure.com", // Azure OpenAI Endpoint
         azureopenaikey);


builder.Plugins.AddFromType<TimePlugin>();
var kernel = builder.Build();

// Use of Core Pluggings

var currentDay = await kernel.InvokeAsync("TimePlugin", "DayOfWeek");
Console.WriteLine(currentDay);
var month = await kernel.InvokeAsync("TimePlugin", "Month");
Console.WriteLine(month);
Console.WriteLine("\nPress any key to continue...\n\n");
Console.ReadKey();


// Example 1 Travel Assistant Language and History


string task = "coding";
string coding_lang = "c#";
string prompt = @$"Create a list of helpful phrases and 
    words in ${task} to start a program in ${coding_lang}";

var result = await kernel.InvokePromptAsync(prompt);
Console.WriteLine(result);

Console.WriteLine("\nPress any key to continue...\n\n");
Console.ReadKey();

string language = "german";
string history = @"I'm traveling with my kids and one of them 
    has a peanut allergy.";

string prompt2 = @$"Consider the traveler's background:
    ${history}

    Create a list of helpful phrases and words in 
    ${language} a traveler would find useful.

    Group phrases by category. Include common direction 
    words. Display the phrases in the following format: 
    Hello - Ciao [chow]";


var result2 = await kernel.InvokePromptAsync(prompt2);
Console.WriteLine(result2);
Console.WriteLine("\nPress any key to continue...\n\n");
Console.ReadKey();
 
// Example3  Planning trip Few Shot

string input = @"I'm planning an anniversary trip with my spouse. We like hiking, mountains, 
    and beaches. Our travel budget is $15000";
string prompt3 = @$"
    The following is a conversation with an AI travel assistant. 
    The assistant is helpful, creative, and very friendly.

    <message role=""user"">Can you give me some travel destination suggestions?</message>

    <message role=""assistant"">Of course! Do you have a budget or any specific 
    activities in mind?</message>

    <message role=""user"">${input}</message>";


var result3 = await kernel.InvokePromptAsync(prompt3);
Console.WriteLine(result3
    );
Console.WriteLine("\nPress any key to continue...\n\n");
Console.ReadKey();

// Example4 Planning trip Few Shot Resume
string input2 = @"I have a vacation from June 1 to July 22. I want to go to Greece. 
    I live in Chicago.";
string prompt4 = @$"
<message role=""system"">Instructions: Identify the from and to destinations 
and dates from the user's request</message>

<message role=""user"">Can you give me a list of flights from Seattle to Tokyo? 
I want to travel from March 11 to March 18.</message>

<message role=""assistant"">Seattle|Tokyo|03/11/2024|03/18/2024</message>

<message role=""user"">${input2}</message>";


var result4 = await kernel.InvokePromptAsync(prompt4);
Console.WriteLine(result4
    );
Console.WriteLine("\nPress any key to continue...\n\n");
Console.ReadKey();

// Example5 Few shot CoT Prompt

string input3 = @"When I was 6, my sister was half of my age. Now I’m 70 years old.
Answer how old is my sister now?";
string prompt5 = @$"
<message role=""system"">Instructions: You are a bot specialized in Maths and Logic. Read the Instruction below and provide an answer. 
Detail the intermediary steps you follow to provide the answer.</message>

<message role=""user"">In this task, you are given an input list A. You need to find all the elements of the list that are numbers and calculate their sum.
Consider the posibility of having negative numbers

['i', 'P', 'h', '849', 'e', '3' ].</message>

<message role=""assistant"">852. Explanation: the only two possible numbers in the list ['i', 'P', 'h', '849', 'e', '3' ] are '849' and '3', they are strings
but they can be converted to numbers. Then the sum of 849 + 3 = 852./message>

<message role=""user"">${input3}</message>";


var result5 = await kernel.InvokePromptAsync(prompt5);
Console.WriteLine(result5
    );
Console.WriteLine("\nPress any key to continue...\n\n");
Console.ReadKey();


// Example 6 Prompts from Files CoT from File


var plugins = kernel.CreatePluginFromPromptDirectory("D:\\repos2\\c#\\plugings\\sk_plugins\\Prompts\\");
string input6 = "When I was 6, my sister was half of my age. Now I’m 70 years old.\r\nAnswer how old is my sister now?";

var result6 = await kernel.InvokeAsync(
    plugins["CoT"],
    new() { { "cotinstruction", input6 } });

Console.WriteLine(result6);
Console.WriteLine("\nPress any key to continue...\n\n");
Console.ReadKey();


// Example 7 Travel agency Prompts from Files

//kernel.ImportPluginFromType<ConversationSummaryPlugin>();
var prompts = kernel.ImportPluginFromPromptDirectory("D:\\repos2\\c#\\plugings\\sk_plugins\\Prompts\\TravelPlugins\\");

ChatHistory history2 = [];
string input7 = @"I'm planning an anniversary trip with my spouse. We like hiking, 
    mountains, and beaches. Our travel budget is $15000";

var result7 = await kernel.InvokeAsync<string>(prompts["SuggestDestinations"],
    new() { { "input", input7 } });

Console.WriteLine(result7);
history2.AddUserMessage(input7);
history2.AddAssistantMessage(result7);

Console.WriteLine("Where would you like to go?");
String input8 = Console.ReadLine().ToString();

var result8 = await kernel.InvokeAsync<string>(prompts["SuggestActivities"],
    new() {
        { "history", history2 },
        { "destination", input8 },
    }
);
Console.WriteLine(result8);