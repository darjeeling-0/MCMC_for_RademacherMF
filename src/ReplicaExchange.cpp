#include "ReplicaExchange.hpp"
#include <iomanip>

MultiCanonical::MultiCanonical(int rep_num, int N, int M, std::vector<double> lambdas, int seed_Data, int seed_gen, double prob)
{
    this->N = N;
    this->M = M;
    this->rep_num = rep_num;
    this->lambdas = lambdas;
    
    replicas.reserve(rep_num);
    gen = std::mt19937_64(seed_gen);
    UNI_double = std::uniform_real_distribution<double> (0.0, 1.0);
    for(int i = 0; i < rep_num; ++i){
        replicas.push_back(MF_Rad(N, M, lambdas[i], seed_Data, gen(), prob));
    }
    replica_exchange_attempt= std::vector<double>(rep_num - 1, 0.0);
    replica_exchange_success = std::vector<double>(rep_num - 1, 0.0);
    CommunicationBarrier = std::vector<double>(rep_num, 0.0);

    FreeEntropy = std::vector<double>(rep_num, 0.0);
    Jackknife_FreeEntropy = std::vector<double>(rep_num, 0.0);
    Jackknife_Variance = std::vector<double>(rep_num, 0.0);
    MI = std::vector<double>(rep_num, 0.0);
    Jackknife_MI = std::vector<double>(rep_num, 0.0);
    Jackknife_MI_Variance = std::vector<double>(rep_num, 0.0);
}

void MultiCanonical::Replica_Exchange(int parity){
    for(int i = parity; i < rep_num - 1; i += 2){
        replica_exchange_attempt[i] += 1;
        int i_adj = i + 1;
        double joint_weight_current = sqrt(lambdas[i] / N) * replicas[i].E_interaction_noise + lambdas[i] / N * replicas[i].E_interaction - 0.5 * lambdas[i] / N * replicas[i].E_norm;
        joint_weight_current += sqrt(lambdas[i_adj] / N) * replicas[i_adj].E_interaction_noise + lambdas[i_adj] / N * replicas[i_adj].E_interaction - 0.5 * lambdas[i_adj] / N * replicas[i_adj].E_norm;

        double joint_weight_proposed = sqrt(lambdas[i] / N) * replicas[i_adj].E_interaction_noise + lambdas[i] / N * replicas[i_adj].E_interaction - 0.5 * lambdas[i] / N * replicas[i_adj].E_norm;
        joint_weight_proposed += sqrt(lambdas[i_adj] / N) * replicas[i].E_interaction_noise + lambdas[i_adj] / N * replicas[i].E_interaction - 0.5 * lambdas[i_adj] / N * replicas[i].E_norm;

        double dE = exp(joint_weight_proposed - joint_weight_current);
        double r = UNI_double(gen);

        if(r < dE){
            replicas[i].state_X , replicas[i_adj].state_X = replicas[i_adj].state_X, replicas[i].state_X;
            replicas[i].E_interaction , replicas[i_adj].E_interaction = replicas[i_adj].E_interaction, replicas[i].E_interaction;
            replicas[i].E_interaction_noise , replicas[i_adj].E_interaction_noise = replicas[i_adj].E_interaction_noise, replicas[i].E_interaction_noise;
            replicas[i].E_norm , replicas[i_adj].E_norm = replicas[i_adj].E_norm, replicas[i].E_norm;

            replicas[i].H = sqrt(lambdas[i] / N) * replicas[i].E_interaction_noise + lambdas[i] / N * replicas[i].E_interaction - 0.5 * lambdas[i] / N * replicas[i].E_norm;
            replicas[i_adj].H = sqrt(lambdas[i_adj] / N) * replicas[i_adj].E_interaction_noise + lambdas[i_adj] / N * replicas[i_adj].E_interaction - 0.5 * lambdas[i_adj] / N * replicas[i_adj].E_norm;

            replica_exchange_success[i] += 1;
        }
    }
}

// Adapt the sequence of lambdas such that the replica swap rejection rate 
// is uniform across all replicas
// ref : https://arxiv.org/abs/1905.02939
void MultiCanonical::Adapt_lambda(){
    CommunicationBarrier[0] = 0.0;
    for(int i = 0; i < rep_num - 1; ++i){
        CommunicationBarrier[i + 1] = CommunicationBarrier[i] + (1.0 - 1.0 * replica_exchange_success[i] / replica_exchange_attempt[i]);
        replica_exchange_attempt[i] = 0.0;
        replica_exchange_success[i] = 0.0;
    }
    double first_deriv0 = ( CommunicationBarrier[1] - CommunicationBarrier[0] ) / (lambdas[1] - lambdas[0]);
    tk::spline CommunicationBarrierSpline;
    CommunicationBarrierSpline.set_boundary(tk::spline::first_deriv, first_deriv0, tk::spline::second_deriv , 0.0);
    CommunicationBarrierSpline.set_points(lambdas, CommunicationBarrier);
    CommunicationBarrierSpline.make_monotonic();
    double Lambda = CommunicationBarrier[rep_num - 1];
    for(int i = 1; i < rep_num - 1; ++i){
        double Lambda_i = 1.0 * i / rep_num * Lambda;
        std::vector<double> a = CommunicationBarrierSpline.solve(Lambda_i);
        replicas[i].lambda = a[0];
        lambdas[i] = a[0];
        replicas[i].Calculate_Energy();
    }
}

// Calculate the Mutual Information using Annealed Importance Sampling
// ref: Simulating normalizing constants: from importance sampling to bridge sampling to path sampling
// Andrew Gelman, Xiao-Li Meng
void MultiCanonical::Calculate_MI(){
    MI = std::vector<double>(rep_num, 0.0);
    for(int i = 0; i < rep_num - 1; ++i){
        int history_num = replicas[i].history_interaction.size();
        double EXP_dH = 0.0;
        double dH0 = replicas[i+1].H0 - replicas[i].H0;

        #pragma omp parallel for reduction(+:EXP_dH) schedule(dynamic)
        for(int n = 0; n < history_num; ++n){
            double dH = (sqrt(lambdas[i+1] / N) - sqrt(lambdas[i] / N)) * replicas[i].history_interaction_noise[n] + 
             (lambdas[i+1] - lambdas[i]) / N * replicas[i].history_interaction[n] - 
             0.5 * (lambdas[i+1] - lambdas[i]) / N * replicas[i].history_norm[n];
            EXP_dH += exp(dH - dH0);
        }
        MI[i + 1] = MI[i] + log(EXP_dH / history_num);
    }
}

void MultiCanonical::Jackknife_MI_estimate(int fold_num){
    // Calculate Biased estimator
    Calculate_MI();
    std::vector<std::vector<double>> SampleSplit_MI(fold_num, std::vector<double>(rep_num, 0.0));
    //Calculate the MI for each fold split
    for(int fold = 0; fold < fold_num; ++fold){
        for(int i = 0; i < rep_num - 1; ++i){
            int history_num = replicas[i].history_interaction.size();
            int start = fold * history_num / fold_num;
            int end = (fold + 1) * history_num / fold_num;
            double EXP_dH = 0.0;
            double dH0 = replicas[i+1].H0 - replicas[i].H0;

            #pragma omp parallel for reduction(+:EXP_dH) schedule(dynamic)
            for(int n = 0 ; n < history_num; ++n){
                if(n >= start && n < end) continue;
                double dH = (sqrt(lambdas[i+1] / N) - sqrt(lambdas[i] / N)) * replicas[i].history_interaction_noise[n] + 
                (lambdas[i+1] - lambdas[i]) / N * replicas[i].history_interaction[n] - 
                0.5 * (lambdas[i+1] - lambdas[i]) / N * replicas[i].history_norm[n];
                EXP_dH += exp(dH - dH0);
            }
            SampleSplit_MI[fold][i + 1] = SampleSplit_MI[fold][i] + log(EXP_dH / (history_num - (end - start)));
        }
    }

    //Calculate the Jackknife estimate using Efron's debiased estimator
    std::vector<double> Jackknife_MI_naive(rep_num, 0.0);
    for(int i = 0; i < rep_num; ++i){
        Jackknife_MI_naive[i] = 0.0;
        for(int fold = 0; fold < fold_num; ++fold){
            Jackknife_MI_naive[i] += SampleSplit_MI[fold][i];
        }
        Jackknife_MI_naive[i] /= fold_num;
        Jackknife_MI[i] = fold_num *  MI[i] - (fold_num - 1) * Jackknife_MI_naive[i];
        Jackknife_MI[i] /= -N * M;
    }
    Jackknife_MI_Variance[0] = 0.0;
    // Calculate the Jackknife estimate of the variance
    for(int i = 1; i < rep_num; ++i){
        Jackknife_MI_Variance[i] =0.0;
        for(int fold = 0; fold < fold_num; ++fold){
            Jackknife_MI_Variance[i] += (SampleSplit_MI[fold][i] - Jackknife_MI_naive[i]) * (SampleSplit_MI[fold][i] - Jackknife_MI_naive[i]);
        }
        Jackknife_MI_Variance[i] *= 1.0 * (fold_num - 1) / fold_num / (1.0 * N * M) / (1.0 * N * M);
        Jackknife_MI_Variance[i] += Jackknife_MI_Variance[i-1];
    }
}