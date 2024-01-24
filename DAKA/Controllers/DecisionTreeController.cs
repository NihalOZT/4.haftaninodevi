using DAKA.Models;
using Microsoft.AspNetCore.Mvc;
using Microsoft.ML;
using Microsoft.ML.Data;
using System.Reflection;

namespace DAKA.Controllers
{
    public class DecisionTreeController : Controller
    {
        private readonly MLContext _mlContext;
        // Readonly bir alanın sadece okunabilir olduğunu belirtir.
        private readonly PredictionEngine<PredictionModel, PredictionResult> _predictionEngine;

        public DecisionTreeController()
        {
            _mlContext = new MLContext();

            // Örnek bir veriyle modeli eğitme.

            var traininData = new[]
            {
                new PredictionModel { Feature1 = 1, Feature2 = 1, Prediction= "A sınıfı"},
                new PredictionModel { Feature1 = 2, Feature2 = 2, Prediction= "B sınıfı"},
                new PredictionModel { Feature1= 3, Feature2 = 3, Prediction = "A sınıfı"},
                new PredictionModel { Feature1= 4, Feature2 = 4, Prediction = "B sınıfı"}
            };
            // Verileri algoritmaya hazırlama 
            var dataView = _mlContext.Data.LoadFromEnumerable(traininData);
            var pipeline = _mlContext.Transforms.Conversion.MapValueToKey("Label", "Prediction")
                .Append(_mlContext.Transforms.Concatenate("Features", "Feature1", "Feature2"))
                .Append(_mlContext.Transforms.Conversion.MapValueToKey("Prediction", "Prediction"))
                .Append(_mlContext.Transforms.Conversion.MapValueToKey("Label", "Label"))
                .Append(_mlContext.Transforms.Conversion.MapValueToKey("Features", "Features"));

            // DT İş akışının belirtilmesi

            var trainer = _mlContext.MulticlassClassification.Trainers.SdcaNonCalibrated();
            // Stochastic Dual Coordinate Ascent-non-Calibrated = ML.NET kütüphanesinde
            // kullanılan ikili sınıflandırma problemini çözen algoritma
            var traininPipeline = pipeline.Append(_mlContext.Transforms.Conversion.MapValueToKey("Prediction", "Prediction"))
                .Append(_mlContext.Transforms.Conversion.MapValueToKey("Label", "Label"))
                .Append(trainer)
                .Append(_mlContext.Transforms.Conversion.MapValueToKey("Label", "Prediction"))
                .Append(_mlContext.Transforms.Conversion.MapValueToKey("Prediction", "Prediction"));


            // modelin hangi verilerle eğitilmesi gerektiğini belirtip eğitilmiş modeli kaydetme işlemi
            var trainedModel = traininPipeline.Fit(dataView);
            _predictionEngine = _mlContext.Model.CreatePredictionEngine<PredictionModel, PredictionResult>(trainedModel);
        }

        public IActionResult Index()
        {
            return View();
        }

        [HttpPost]
        public IActionResult PredictionResult(PredictionModel input)
        {
            var prediction = _predictionEngine.Predict(input);
            return View("PredictionResult", prediction);
        }
    }

   
}
