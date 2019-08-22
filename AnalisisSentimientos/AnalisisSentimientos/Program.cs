using System;
using Microsoft.ML.Models;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using static AnalisisSentimientos.Sentimiento;

namespace AnalisisSentimientos
{
    class Program
    {
        /// <summary>
        /// Objetivo: Desarrollar una aplicacion de ML que nos indique si los comentarios de los usuarios son positivos o negativos
        /// Cómo: Algoritmo de clasificación binaria.
        /// Datos: Archivo con datos de entrenamiento + archivos con datos de prueba.
        /// </summary>
        const string _rutaDatosEntrenamiento = @"..\..\data\sentiment labelled sentences\imdb_labelled.txt";
        const string _rutaDatosPrueba = @"..\..\data\sentiment labelled sentences\yelp_labelled.txt";
        static void Main(string[] args)
        {
            var modelo = EntrenayPredice();
            Evalua(modelo);
            Console.ReadLine();
        }

        public static PredictionModel<DatosSentimiento, PrediccSentimiento> EntrenayPredice()
        {
            //Recoleccion de Datos
            var pipeline = new LearningPipeline();
            pipeline.Add(new TextLoader<DatosSentimiento>(_rutaDatosEntrenamiento, useHeader: false, separator: "tab"));
            pipeline.Add(new TextFeaturizer("Features", "Texto"));
            pipeline.Add(new FastTreeBinaryClassifier() { NumLeaves = 5, NumTrees = 5, MinDocumentsInLeafs = 2 }); //Aprediz de arbol de decision (Regresión);

            //Entrenamiento
            PredictionModel<DatosSentimiento, PrediccSentimiento> modelo = pipeline.Train<DatosSentimiento, PrediccSentimiento>();

            IEnumerable<DatosSentimiento> sentimientos = new[]
            {
                new DatosSentimiento
                {
                    Texto="This movie was very boring",
                    Etiqueta = 0
                },

                 new DatosSentimiento
                {
                    Texto="The movie does not have a great story to tell.",
                    //Texto="The movie did not get my attention",
                    Etiqueta = 0
                },
                  new DatosSentimiento
                {
                    //Texto="A super exciting and entertaining movie",
                    Texto="It was a very exciting from the start",
                    Etiqueta = 1
                }
            };

            var predicciones = modelo.Predict(sentimientos);
            Console.WriteLine();
            Console.WriteLine("Predicción de sentimientos");
            Console.WriteLine("---------------------------");

            var sentimientosYpredicciones = sentimientos.Zip(predicciones, (sent, predict) => (sent, predict));

            foreach (var item in sentimientosYpredicciones)
            {
                Console.WriteLine($"Sentimiento: {item.sent.Texto} | Predicción: {(item.predict.Etiqueta ? "Positivo: )": "Negativo: (")}");
            }
            Console.WriteLine();

            //Evaluación del Modelo
            return modelo;
        }

        public static void Evalua(PredictionModel<DatosSentimiento, PrediccSentimiento> modelo)
        {
            var datosPrueba = new TextLoader<DatosSentimiento>(_rutaDatosPrueba, useHeader: false, separator: "tab");
            var evaluador = new BinaryClassificationEvaluator();//Obtener Metricas de evaluacion
            BinaryClassificationMetrics metricas = evaluador.Evaluate(modelo, datosPrueba); //modelo:es el modelo de prediccion entrenado que vamos a evaluar

            Console.WriteLine();
            Console.WriteLine("Evaluación de métricas de calidad del Modelo de Predicción");
            Console.WriteLine("---------------------------------");
            Console.WriteLine($"Precisión: {metricas.Accuracy:P2}"); //La presicion indica que tan acertado ha sido el algoritmo durante la prediccion
            Console.WriteLine($"AUC: {metricas.Auc:P2}"); //Medida del rendimiento para´problemas de clasificacion binaria (1.0 correcto)
            Console.WriteLine($"Log-loss: {metricas.LogLoss:P2}");
            Console.WriteLine($"F1SCore: {metricas.F1Score:P2}");
            //AUC=78%-> de cada 100 elementos se han clasificado 78 correctamente

        }
    }
}
