#ifndef REPLICA_EXCHANGE
#define REPLICA_EXCHANGE

#include "MF_Rad.hpp"
#include "spline.h"

class MultiCanonical{
    public:
    int rep_num, N, M, overlap; 
    std::vector<double> FreeEntropy, Jackknife_FreeEntropy, Jackknife_Variance;
    std::vector<double> MI, Jackknife_MI, Jackknife_MI_Variance;
    std::vector<double> lambdas, replica_exchange_success, replica_exchange_attempt;
    std::vector<MF_Rad> replicas;
    std::mt19937_64 gen;
    std::vector<double> CommunicationBarrier;
    std::uniform_real_distribution<double> UNI_double;
    void Replica_Exchange(int parity);
    void Adapt_lambda();
    MultiCanonical(int rep_num, int N, int M, std::vector<double> lambdas, int seed_Data, int seed_gen, double prob);
    void Calculate_Free_Entropy();
    void Calculate_MI();
    void Calculate_overlap();
    void Jackknife_MI_estimate(int fold_num);
};


#endif // REPLICA_EXCHANGE