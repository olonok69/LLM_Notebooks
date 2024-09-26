// Copyright (c) Microsoft. All rights reserved.

using SkiaSharp;
using System.Net.Http;

// ReSharper disable InconsistentNaming
namespace classification.config;
public static class SkiaUtils
{
    // Function used to display images in the notebook
    public static async Task SaveImage(string url, int width, int height, string outputPath)
    {
        SKImageInfo info = new SKImageInfo(width, height);
        SKSurface surface = SKSurface.Create(info);
        SKCanvas canvas = surface.Canvas;
        canvas.Clear(SKColors.White);
        var httpClient = new HttpClient();
        using (Stream stream = await httpClient.GetStreamAsync(url))
        using (MemoryStream memStream = new MemoryStream())
        {
            await stream.CopyToAsync(memStream);
            memStream.Seek(0, SeekOrigin.Begin);
            SKBitmap webBitmap = SKBitmap.Decode(memStream);
            
            using var streamf = new FileStream(outputPath, FileMode.Create, FileAccess.Write);
            using var image = SKImage.FromBitmap(webBitmap);
            using var encodedImage = image.Encode();
            encodedImage.SaveTo(streamf);
            //canvas.DrawBitmap(webBitmap, 0, 0, null);
            //surface.Draw(canvas, 0, 0, null);
        };
        //surface.Snapshot().Display();
    }
}
