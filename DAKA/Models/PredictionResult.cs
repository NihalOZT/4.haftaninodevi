using Microsoft.ML.Data;

namespace DAKA.Models
{
    public class PredictionResult
    {
        [ColumnName("PredictionLabel")]
        public string Prediction { get; set; }
    }
}
