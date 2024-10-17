#ifndef MF_RAD
#define MF_RAD 

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <functional>
#include <random>
#include <chrono>
#include <fstream>
#include <cstdint>
#include <bitset>
#include <cassert>
#include <set>

typedef uint64_t spin;
typedef std::vector<std::vector<double>> Mat;
extern const int NMC_MAX;

inline void Flip_spin(spin &s, int i)
{
    s ^= 1ULL << (i);
}

inline int countOneBits(spin n) {
    int count = 0;
    while (n) {
        n &= (n - 1);
        count++;
    }
    return count;
}

inline int spin_inner_product(const spin& spin1, const spin& spin2, int M) {
    return 2 * countOneBits(~ (spin1 ^ spin2)) -128 + M;
}

inline int isNthBitSet(int_fast64_t num, int n) {
    return ((num & (1ULL << n)) != 0);
}

inline std::vector<spin> transposeMatrix(const std::vector<spin> &M, int num_rows, int num_cols){
    std::vector<spin> result(num_cols, 0);
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            if (M[i] & (1ULL << j)) {
                result[j] |= (1ULL << i);
            }
        }
    }
    return result;
}


class MF_Rad
{
    public:
    MF_Rad(int N, int M, double lambda, int seed_Data, int seed_gen, double prob);
    void Rerandomize(double prob);
    std::mt19937_64 gen, gen_Data;
    int N, M;
    bool start_record ;
    std::vector<spin> state_X; 
    std::vector<spin> X0; 
    std::vector<int> index_row, index_col;
    std::vector<std::vector<spin>> history_X;
    std::vector<double> history_interaction, history_interaction_noise, history_norm, history_H;
    std::vector<std::vector<std::vector<int>>> history_overlap;
    std::vector<std::vector<int>> X0_inner_product;
    std::uniform_int_distribution<spin> UNI;
    std::uniform_real_distribution<double> UNI_double;
    std::normal_distribution<double> NORMAL;
    Mat Y; 
    Mat Z;
    double lambda;
    double E_interaction, E_interaction_noise, E_norm, H, H0, MI_p;
    double E0_interaction, E0_interaction_noise, E0_norm;
    double average_MMSE, variance_MMSE, average_MMSE_, Jackknife_MMSE, Jackknife_MMSE_Variance;
    Mat average_overlap, average_overlap_sq;
    Mat average_self_overlap, average_self_overlap_sq;
    std::vector<double> sorted_overlap_sq;
    Mat X0_overlap;
    Mat X0_overlap_sq;
    Mat S;
    void Calculate_average_overlap();
    void Record_overlap();
    void Calculate_average_MMSE();
    void Calculate_Energy();
    void Delta_Energy(int i, int mu, double &dE_interaction, double &dE_interaction_noise, double &dE_norm);
    void MonteCarloSweep(bool record_or_not);
    void MonteCarloStep(int i, int mu);
};

#endif // MF_RAD
