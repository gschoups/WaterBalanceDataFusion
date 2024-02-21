using System;
using System.Collections.Generic;

using Microsoft.ML.Probabilistic;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Models;

namespace WaterBalanceDataFusion;

using GaussianArrayEstimator = ArrayEstimator<GaussianEstimator, DistributionStructArray<Gaussian, double>, Gaussian>;
using Range = Microsoft.ML.Probabilistic.Models.Range;

public partial class WaterBalanceDataFusion_EP : IGeneratedAlgorithm
{
    public int NumberOfIterationsEP { get; set; } = 3;
    public int NumberOfIterationsPerParameterMCMC { get; set; } = 2000;
    public VectorGaussian JointParameterPosterior { get; private set; }

    /// <summary>
    /// Runs MCMC and computes posteriors of error parameters and water balance variables.
    /// </summary>
    /// <returns>Largest value of the unnormalized log parameter posterior density.</returns>
    public double InferMCMC()
    {
        int nIter = NumberOfIterationsPerParameterMCMC * ParPrior.Dimension;
        int nInitial = 100 * ParPrior.Dimension;
        var samples = new List<Vector>(nIter + nInitial);
        var logProbs = new List<double>(nIter + nInitial);
        for (int i = 0; i < nInitial; i++)
        {
            var par = ParPrior.Sample();
            samples.Add(par);
            logProbs.Add(GetLogProb(par));
        }
        DEMC_Z(GetLogProb, nIter, samples, logProbs);
        EstimateLatentMarginals(samples);
        int mode = IndexOfMax(logProbs);
        return logProbs[mode];
    }

    /// <summary>
    /// Evaluates the unnormalized log parameter posterior (Eq. 17) at a specific error parameter vector.
    /// </summary>
    /// <param name="par">The error parameter vector.</param>
    /// <returns>The unnormalized log parameter posterior density.</returns>
    private double GetLogProb(Vector par)
    {
        Par = par;
        Execute(NumberOfIterationsEP);
        return EvidenceMarginal().LogOdds;
    }

    /// <summary>
    /// Computes error parameter and water balance posteriors (Eq. 19) given a list of error parameter vectors sampled from the posterior.
    /// </summary>
    /// <param name="pars">The error parameter vectors.</param>
    private void EstimateLatentMarginals(List<Vector> pars)
    {
        var estPar = new VectorGaussianEstimator(ParPrior.Dimension);

        var estwE = new BetaEstimator();
        var estfE = new GammaEstimator();
        var estrE = new BetaEstimator();
        var estA = new GammaEstimator();
        var estDelta = new BetaEstimator();
        var estSStd = new GammaEstimator();
        var estaQ = new GammaEstimator();
        var estbQ = new GammaEstimator();
        var estwP = new BetaEstimator();
        var estrP = new BetaEstimator();
        var estaI = new GammaEstimator();

        var estP = new GaussianArrayEstimator(Nt, i => new GaussianEstimator());
        var estE = new GaussianArrayEstimator(Nt, i => new GaussianEstimator());
        var estQ = new GaussianArrayEstimator(Nt, i => new GaussianEstimator());
        var estI = new GaussianArrayEstimator(Nt, i => new GaussianEstimator());
        var estS = new GaussianArrayEstimator(Nt, i => new GaussianEstimator());
        var estS0 = new GaussianEstimator();

        double nLast = 0.2 * pars.Count;//compute posterior from last 20% samples
        for (int i = 0; i < nLast; i++)
        {
            int index = i;// Rand.Int(0, 1000);
            Par = pars[pars.Count - 1 - index];

            Execute(NumberOfIterationsEP);

            estPar.Add(Par);

            estwE.Add(wE_marginal_F);
            estfE.Add(fE_marginal_F);
            estrE.Add(rE_marginal_F);
            estA.Add(A_marginal_F);
            estDelta.Add(Delta_marginal_F);
            estSStd.Add(SStd_marginal_F);
            estaQ.Add(aQ_marginal_F);
            estbQ.Add(bQ_marginal_F);
            estwP.Add(wP_marginal_F);
            estrP.Add(rP_marginal_F);
            estaI.Add(aI_marginal_F);

            estP.Add(P_marginal_F);
            estE.Add(E_marginal_F);
            estQ.Add(Q_marginal_F);
            estI.Add(I_marginal_F);
            estS.Add(S_marginal_F);
            estS0.Add(S0_marginal_F);
        }

        JointParameterPosterior = estPar.GetDistribution(new VectorGaussian(ParPrior.Dimension));

        wE_marginal_F = estwE.GetDistribution(wE_marginal_F);
        fE_marginal_F = estfE.GetDistribution(fE_marginal_F);
        rE_marginal_F = estrE.GetDistribution(rE_marginal_F);
        A_marginal_F = estA.GetDistribution(A_marginal_F);
        Delta_marginal_F = estDelta.GetDistribution(Delta_marginal_F);
        SStd_marginal_F = estSStd.GetDistribution(SStd_marginal_F);
        aQ_marginal_F = estaQ.GetDistribution(aQ_marginal_F);
        bQ_marginal_F = estbQ.GetDistribution(bQ_marginal_F);
        wP_marginal_F = estwP.GetDistribution(wP_marginal_F);
        rP_marginal_F = estrP.GetDistribution(rP_marginal_F);

        P_marginal_F = estP.GetDistribution(P_marginal_F);
        E_marginal_F = estE.GetDistribution(E_marginal_F);
        Q_marginal_F = estQ.GetDistribution(Q_marginal_F);
        S_marginal_F = estS.GetDistribution(S_marginal_F);
        S0_marginal_F = estS0.GetDistribution(S0_marginal_F);
    }

    /// <summary>
    /// Generates EP inference code for computing conditional water balance posteriors.
    /// </summary>
    public static void GenerateCode()
    {
        // Dummy input values
        int ntValue = 1;
        var value = new double[ntValue];
        var parPriorValue = new VectorGaussian(Vector.Zero(11), PositiveDefiniteMatrix.Identity(11));
        var s0PriorValue = new Gaussian(0, 1);

        // Define variables
        Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
        Variable<int> nt = Variable.Observed(ntValue).Named("Nt");
        Range timeInterval = new(nt);
        Variable<VectorGaussian> parPrior;
        Variable<Gaussian> S0Prior;
        VariableArray<double> PObs1, PObs2, PStd, EObs1, EObs2, QObs, QObsVar, IObs, SObs;
        VariableArray<double> P, E, Q, I, S;
        Variable<double> S0, wE, fE, rE, A, Delta, SStd, aQ, bQ, wP, rP, aI;
        Variable<Vector> par;

        // Define model
        using (Variable.If(evidence))
        {
            // Parameters
            parPrior = Variable.Observed(parPriorValue).Named("ParPrior");
            par = Variable.Random<Vector, VectorGaussian>(parPrior).Named("Par");
            wE = Variable.Logistic(Variable.GetItem(par, 0)).Named("wE");
            fE = Variable.Exp(Variable.GetItem(par, 1)).Named("fE");
            rE = Variable.Logistic(Variable.GetItem(par, 2)).Named("rE");
            A = Variable.Exp(Variable.GetItem(par, 3)).Named("A");
            Delta = Variable.Logistic(Variable.GetItem(par, 4)).Named("Delta");
            SStd = Variable.Exp(Variable.GetItem(par, 5)).Named("SStd");
            aQ = Variable.Exp(Variable.GetItem(par, 6)).Named("aQ");
            bQ = Variable.Exp(Variable.GetItem(par, 7)).Named("bQ");
            wP = Variable.Logistic(Variable.GetItem(par, 8)).Named("wP");
            rP = Variable.Logistic(Variable.GetItem(par, 9)).Named("rP");
            aI = Variable.Exp(Variable.GetItem(par, 10)).Named("aI");

            // Observations
            PObs1 = Variable.Observed(value, timeInterval).Named("PObs1");
            PObs2 = Variable.Observed(value, timeInterval).Named("PObs2");
            PStd = Variable.Observed(value, timeInterval).Named("PStd");
            EObs1 = Variable.Observed(value, timeInterval).Named("EObs1");
            EObs2 = Variable.Observed(value, timeInterval).Named("EObs2");
            QObs = Variable.Observed(value, timeInterval).Named("QObs");
            QObsVar = Variable.Observed(value, timeInterval).Named("QObsVar");
            IObs = Variable.Observed(value, timeInterval).Named("IObs");
            SObs = Variable.Observed(value, timeInterval).Named("SObs");
            S0Prior = Variable.Observed(s0PriorValue).Named("S0Prior");

            // Water balance variables
            P = Variable.Array<double>(timeInterval).Named("P");
            E = Variable.Array<double>(timeInterval).Named("E");
            Q = Variable.Array<double>(timeInterval).Named("Q");
            I = Variable.Array<double>(timeInterval).Named("I");
            S = Variable.Array<double>(timeInterval).Named("S");
            S0 = Variable.Random<double, Gaussian>(S0Prior).Named("S0");

            // Time loop
            using (var time = Variable.ForEach(timeInterval))
            {
                var t = time.Index;

                // P
                var mP = (1 - wP) * PObs1[t] + wP * PObs2[t];
                var sP = Variable.Max(PStd[t], rP * 0.5 * Abs(PObs1[t] - PObs2[t]));
                P[t] = Variable.GaussianFromMeanAndVariance(mP, sP * sP);
                Variable.ConstrainPositive(P[t]);

                // E
                var mE = fE * ((1 - wE) * EObs1[t] + wE * EObs2[t]);
                var sE = Variable.Max(0.1 * mE, rE * 0.5 * Abs(EObs1[t] - EObs2[t]));
                E[t] = Variable.GaussianFromMeanAndVariance(mE, sE * sE);
                Variable.ConstrainPositive(E[t]);

                // Q
                var mQ = Variable.GaussianFromMeanAndVariance(QObs[t], QObsVar[t]);
                var sQ = aQ * QObs[t] + bQ;
                Q[t] = Variable.GaussianFromMeanAndVariance(mQ, sQ * sQ);
                Variable.ConstrainPositive(Q[t]);

                // I (water imports)
                var sI = aI * IObs[t];
                I[t] = Variable.GaussianFromMeanAndVariance(IObs[t], sI * sI);
                Variable.ConstrainPositive(I[t]);

                // S: water balance
                using (Variable.If(t == 0))
                {
                    S[t] = S0 + P[t] - E[t] - Q[t] + I[t];
                }
                using (Variable.If(t > 0))
                {
                    S[t] = S[t - 1] + P[t] - E[t] - Q[t] + I[t];
                }

                // SObs
                var missingSObs = IsNaN(SObs[t]);
                using (Variable.IfNot(missingSObs))
                {
                    const double omega = 2 * Math.PI;
                    var mS = S[t] + A * Sin(omega * (Variable.Double(t) / 12 - Delta));
                    var sS = SStd;
                    SObs[t] = Variable.GaussianFromMeanAndVariance(mS, sS * sS);
                }
            }
        }
        SObs.ObservedValue = value;
        //par.AddAttribute(new PointEstimate());//for optimization (not working atm)
        par.ObservedValue = parPriorValue.GetMean();//for use with MCMC

        // Generate inference code
        var engine = new InferenceEngine();
        engine.OptimiseForVariables = [P, E, Q, I, S, S0, evidence, wE, fE, rE, A, Delta, SStd, aQ, bQ, wP, rP, aI];
        engine.ModelName = "WaterBalanceDataFusion";
        engine.ModelNamespace = "WaterBalanceDataFusion";
        engine.Compiler.GeneratedSourceFolder = "../../../";
        engine.Compiler.WriteSourceFiles = true;
        engine.Compiler.UseSerialSchedules = true;
        engine.Compiler.ShowProgress = false;
        engine.Compiler.OptimiseInferenceCode = true;
        engine.Compiler.CompilerChoice = Microsoft.ML.Probabilistic.Compiler.CompilerChoice.Roslyn;
        engine.Infer(evidence);//this line generates the inference code (and runs inference with the dummy input values)
    }

    // Helper functions
    private static Variable<bool> IsNaN(Variable<double> x) => Variable<bool>.Factor(double.IsNaN, x);
    private static Variable<double> Sin(Variable<double> x) => Variable<double>.Factor(Math.Sin, x);
    private static Variable<double> Abs(Variable<double> x) => Variable<double>.Factor(Math.Abs, x);

    // Returns index of the (first occurrence) of the maximum value in a list of values.
    private static int IndexOfMax(List<double> values)
    {
        double maxValue = double.NegativeInfinity;
        int index = -1;
        for (int i = 0; i < values.Count; i++)
        {
            if (maxValue < values[i])
            {
                maxValue = values[i];
                index = i;
            }
        }
        return index;
    }

    /// <summary>
    /// Single-chain no-snooker version of the differential evolution MCMC algorithm of ter Braak and Vrugt (2008) to generate samples from a d-dimensional density p(x).
    /// </summary>
    /// <remarks>
    /// Reference: ter Braak, C. J., and Vrugt, J. A. (2008). Differential evolution Markov chain with snooker updater and fewer chains. 
    /// Statistics and Computing, 18(4), 435-446.
    /// </remarks>
    /// <param name="f">Function that computes log-density value of target distribution p(x).</param>
    /// <param name="N">Number of samples to generate.</param>
    /// <param name="X">List with 10*d initial values for x. Generated samples will be added to this list.</param>
    /// <param name="F">List with 10*d corresponding log-density values. Log-density values of samples in X will be added to this list.</param>
    private static void DEMC_Z(Func<Vector, double> f, int N, List<Vector> X, List<double> F, double probGamma1 = 0.2)
    {
        int d = X[0].Count;
        double normalJump = 2.38 / Math.Sqrt(2 * d);
        for (int t = 0; t < N; t++)
        {
            // STEP 1: generate a new sample using differential evolution
            // 1a. randomly select two different indices i and j into X (excluding last index)
            int count = X.Count;
            int i = Rand.Int(count - 1);//excludes count-1 !
            int j = i;
            while (j == i) j = Rand.Int(count - 1);//excludes count-1 !

            // 1b. randomly set jumping rate gamma to 1.0 (10% chance) or to 2.38/sqrt(2*d) (90% chance)
            double gamma = (Rand.Double() < probGamma1) ? 1 : normalJump;

            // 1c. propose new sample using differential evolution
            var xOld = X[count - 1];
            var xNew = Vector.Zero(d);
            for (int k = 0; k < d; k++) xNew[k] = xOld[k] + gamma * (X[i][k] - X[j][k]) + 0.0001 * Rand.Normal();

            // STEP 2: accept the new sample with probability alpha = min(1, p(xNew)/p(xOld))
            double fOld = F[count - 1];
            double fNew = f(xNew);
            double logAlpha = fNew - fOld;
            double u = Rand.Double();
            if (logAlpha > Math.Log(u))//accept the new sample
            {
                X.Add(xNew);
                F.Add(fNew);
            }
            else//reject (copy the old one)
            {
                xNew.SetTo(xOld);
                X.Add(xNew);
                F.Add(fOld);
            }
        }
    }
}