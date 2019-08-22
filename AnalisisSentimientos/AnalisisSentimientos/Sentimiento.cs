using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations.Schema;
using System.Text;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;

namespace AnalisisSentimientos
{
    class Sentimiento
    {
        public class DatosSentimiento
        {
            [Microsoft.ML.Runtime.Api.Column(ordinal: "0")]
            public string Texto;
            [Microsoft.ML.Runtime.Api.Column(ordinal: "1", name:"Label")]
            public float Etiqueta;
        }

        public class PrediccSentimiento
        {
            [ColumnName("PredictedLabel")]
            public bool Etiqueta;
        }
    }
}
