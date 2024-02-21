using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using System.Globalization;

using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Utilities;

namespace WaterBalanceDataFusion;

class Program
{
    static void Main(string[] args)
    {
        // Enforce dot as decimal separator
        CultureInfo.DefaultThreadCurrentCulture = CultureInfo.InvariantCulture;

        // Uncomment this line to regenerate the EP inference code (only necessary if changes are made to the model)
        //WaterBalanceDataFusion_EP.GenerateCode();

        // Specify basin folder containing data files
        //args = ["../../../basins/Mond"];
        if (args.Length == 0) throw new ArgumentException("Missing basin data folder: dotnet run <basinFolder>");
        string basinFolder = Path.EndsInDirectorySeparator(args[0]) ? args[0] : $"{args[0]}{Path.DirectorySeparatorChar}";

        // Run analysis
        InferWaterBalance(basinFolder);
    }

    private static void InferWaterBalance(string basinFolder)
    {
        // Data - missing observations should be stored as double.NaN
        LoadData(basinFolder, out double[][] PObs, out double[] PStd, out double[][] EObs, out double[] QObs, out double[][] SObs, out double[] IObs);

        // Estimate monthly QObs distributions
        var qEst = Util.ArrayInit(12, m => new MeanVarianceAccumulator());
        for (int t = 0; t < QObs.Length; t++)
        {
            int m = t % 12;
            if (!double.IsNaN(QObs[t])) qEst[m].Add(QObs[t]);
        }

        // Handle missing QObs = NaN
        var QObsVar = new double[QObs.Length];
        for (int t = 0; t < QObs.Length; t++)
        {
            if (double.IsNaN(QObs[t]))
            {
                int m = t % 12;
                QObs[t] = qEst[m].Mean;
                QObsVar[t] = qEst[m].Count == 0 ? double.PositiveInfinity : qEst[m].Variance;
            }
        }

        // Prior for transformed par = (wE, fE, rE, A, Delta, SStd, aQ, bQ, fP, fStd)
        static Gaussian FlatLogitnormal() => new Gaussian(0, 1.4 * 1.4);
        static Gaussian LognormalFromModeAndCV(double mode, double cv)
        {
            double m = Math.Log(mode);
            double v = MMath.Log1Plus(cv * cv);
            return Gaussian.FromMeanAndVariance(m, v) * Gaussian.FromNatural(1, 0);
        }
        var parPrior = new[]
        {
            FlatLogitnormal(),//wE
            LognormalFromModeAndCV(1, 0.5),//fE
            FlatLogitnormal(),//rE
            LognormalFromModeAndCV(30, 2),//A
            FlatLogitnormal(),//delta
            LognormalFromModeAndCV(10, 2),//SStd
            LognormalFromModeAndCV(0.1, 0.01),//aQ, for tight prior use cv=0.01, for vague prior use cv=0.9
            LognormalFromModeAndCV(0.001, 0.01),//bQ
            FlatLogitnormal(),//wP
            FlatLogitnormal(), //rP
            LognormalFromModeAndCV(0.2, 0.01)//aI, aI=0.2 means 20% error on water imports, e.g.
        };
        var priorMean = Vector.Zero(parPrior.Length);
        var priorVariance = new PositiveDefiniteMatrix(parPrior.Length, parPrior.Length);
        for (int i = 0; i < parPrior.Length; i++)
        {
            priorMean[i] = parPrior[i].GetMean();
            priorVariance[i, i] = parPrior[i].GetVariance();
        }

        // Posteriors
        var wb = new WaterBalanceDataFusion_EP
        {
            Nt = PObs[0].Length,
            PObs1 = PObs[0],
            PObs2 = PObs[1],
            PStd = PStd,
            EObs1 = EObs[0],
            EObs2 = EObs[1],
            QObs = QObs,
            IObs = IObs,
            QObsVar = QObsVar,
            SObs = SObs[0],
            S0Prior = Gaussian.FromMeanAndVariance(0, 200 * 200),
            ParPrior = VectorGaussian.FromMeanAndVariance(priorMean, priorVariance)
        };
        double maxLogPosterior = wb.InferMCMC();
        Console.WriteLine($"maxLogPosterior = {maxLogPosterior}");

        // Write results
        WriteResult($"{basinFolder}resultsP.out", wb.PMarginal());
        WriteResult($"{basinFolder}resultsE.out", wb.EMarginal());
        WriteResult($"{basinFolder}resultsQ.out", wb.QMarginal());
        WriteResult($"{basinFolder}resultsI.out", wb.IMarginal());
        WriteResult($"{basinFolder}resultsS.out", wb.SMarginal());
        WriteResult($"{basinFolder}resultsPar.out", new object[] { wb.wE_marginal_F, wb.fE_marginal_F, wb.rE_marginal_F, wb.A_marginal_F, wb.Delta_marginal_F,
                                                                   wb.SStd_marginal_F, wb.aQ_marginal_F, wb.bQ_marginal_F, wb.wP_marginal_F, wb.rP_marginal_F,
                                                                   wb.aI_marginal_F });
    }

    private static void WriteResult<T>(string fileName, IEnumerable<T> items)
    {
        using var file = new StreamWriter(fileName);
        foreach (var item in items) file.WriteLine(item.ToString());
    }

    private static void LoadData(string basinFolder, out double[][] P, out double[] PStd, out double[][] E, out double[] Q, out double[][] S, out double[] I)
    {
        // Read fileNames.txt to get names of the data files
        var fileNames = new Dictionary<string, string>();
        using (StreamReader file = new($"{basinFolder}fileNames.txt"))
        {
            char[] delim = ['\t', ' ', ','];
            while (!file.EndOfStream)
            {
                var line = file.ReadLine();
                var values = line.Split(delim, StringSplitOptions.RemoveEmptyEntries);
                fileNames.Add(values[0], values[1]);
            }
        }

        // Read data files
        I = File.ReadAllLines($"{basinFolder}{fileNames["IObs"]}").Select(double.Parse).ToArray();//mm/month
        Q = File.ReadAllLines($"{basinFolder}{fileNames["QObs"]}").Select(double.Parse).ToArray();//mm/month
        P = new double[2][];
        P[0] = File.ReadAllLines($"{basinFolder}{fileNames["PObs1"]}").Select(double.Parse).ToArray();//mm/month
        P[1] = File.ReadAllLines($"{basinFolder}{fileNames["PObs2"]}").Select(double.Parse).ToArray();//mm/month
        PStd = File.ReadAllLines($"{basinFolder}{fileNames["PStd"]}").Select(double.Parse).ToArray();//mm/month
        E = new double[2][];
        E[0] = File.ReadAllLines($"{basinFolder}{fileNames["EObs1"]}").Select(double.Parse).ToArray();//mm/month
        E[1] = File.ReadAllLines($"{basinFolder}{fileNames["EObs2"]}").Select(double.Parse).ToArray();//mm/month
        S = new double[1][];
        S[0] = File.ReadAllLines($"{basinFolder}{fileNames["SObs"]}").Select(double.Parse).ToArray();//mm

        // Set missing observations to NaN
        for (int t = 0; t < P[0].Length; t++)
        {
            if (Q[t] < 0) Q[t] = double.NaN;
            if (P[0][t] < 0) P[0][t] = double.NaN;
            if (P[1][t] < 0) P[1][t] = double.NaN;
            if (PStd[t] < 0) PStd[t] = double.NaN;
            if (E[0][t] < 0) E[0][t] = double.NaN;
            if (E[1][t] < 0) E[1][t] = double.NaN;
            if (S[0][t] == -99999) S[0][t] = double.NaN;
            if (I[t] == 0) I[t] = 1e-6;
        }
    }
}