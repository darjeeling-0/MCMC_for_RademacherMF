#include "MF_Rad.hpp"

const int NMC_MAX = 1000000;

MF_Rad::MF_Rad(int N, int M, double lambda, int seed_Data, int seed_gen, double prob)
{
    this->N = N;
    this->M = M;
    this->lambda = lambda;
    gen = std::mt19937_64(seed_gen);
    gen_Data = std::mt19937_64(seed_Data);
    state_X.resize(N);
    X0.resize(N);
    history_X.reserve(NMC_MAX);
    history_interaction.reserve(NMC_MAX);
    history_interaction_noise.reserve(NMC_MAX); 
    history_norm.reserve(NMC_MAX);
    history_H.reserve(NMC_MAX);
    Y.resize(N, std::vector<double>(N, 0.0));
    Z.resize(N, std::vector<double>(N, 0.0));
    X0_inner_product.resize(N, std::vector<int>(N, 0));    
    UNI = std::uniform_int_distribution<spin> (0, spin((1ULL << M) - 1));
    UNI_double = std::uniform_real_distribution<double> (0.0, 1.0);
    Rerandomize(prob);  
    Calculate_Energy();
    start_record = false;
    index_row = std::vector<int>(N);
    index_col = std::vector<int>(M);
    for(int i = 0; i < N; ++i){
        index_row[i] = i;
    }
    for(int i = 0; i < M; ++i){
        index_col[i] = i;
    }
}


void MF_Rad::Rerandomize(double prob)
{
    for (int i = 0; i < N; i++)
    {
        X0[i] = UNI(gen_Data);
        //Initialize the state_X to be the same as X0
        state_X[i] = X0[i];
        //Flip the bits of state_X[i] with probability prob
        for (int j = 0; j < M; j++){
            if (UNI_double(gen) < prob) Flip_spin(state_X[i], j);
        }

        for(int j = 0; j < i; ++j){
            X0_inner_product[i][j] = spin_inner_product(X0[i], X0[j], M);
            X0_inner_product[j][i] = X0_inner_product[i][j];
            Z[i][j] = NORMAL(gen_Data);
            Z[j][i] = Z[i][j];
            Y[i][j] = sqrt(lambda / N) * X0_inner_product[i][j] + Z[i][j];
            Y[j][i] = Y[i][j];
        }
        Y[i][i] = 0.0;
        Z[i][i] = 0.0;
    }
}

void MF_Rad::Calculate_Energy(){
    E_interaction = 0.0;
    E_interaction_noise = 0.0;
    E_norm = 0.0;
    E0_interaction = 0.0;
    E0_interaction_noise = 0.0;
    E0_norm = 0.0;
    MI_p = 0.0;
    H0 = 0.0;
    for(int i = 0; i < N; ++i){
        for(int j = 0; j < i; ++j){
            int inner = spin_inner_product(state_X[i], state_X[j], M);
            E_interaction += inner * X0_inner_product[i][j];
            E_interaction_noise += inner * Z[i][j];
            E_norm += inner * inner;
            MI_p += X0_inner_product[i][j] * X0_inner_product[i][j];

            E0_interaction += X0_inner_product[i][j] * X0_inner_product[i][j];
            E0_interaction_noise += Z[i][j] * X0_inner_product[i][j];
            E0_norm += X0_inner_product[i][j] * X0_inner_product[i][j];
        }
    }
    H = sqrt(lambda / N) * E_interaction_noise + lambda / N * E_interaction - 0.5 * lambda / N * E_norm;
    H0 = sqrt(lambda / N) * E0_interaction_noise + lambda / N * E0_interaction - 0.5 * lambda / N * E0_norm;
}

// Return the energy difference if we flip the mu-th bit of the i-th spin, E_new = E_old + dE
void MF_Rad::Delta_Energy(int i, int mu, double& dE_interaction, double& dE_interaction_noise, double& dE_norm){
    dE_interaction = 0.0;
    dE_interaction_noise = 0.0;
    dE_norm = 0.0;
    for(int j = 0; j < N; ++j){
        int sj = (2 * isNthBitSet(state_X[j], mu)  - 1);
        dE_interaction += sj * X0_inner_product[i][j];
        dE_interaction_noise += sj * Z[i][j];
        if(j != i){
            dE_norm += sj * spin_inner_product(state_X[i], state_X[j], M);
        }
    }
    int si = (2 * isNthBitSet(state_X[i], mu) - 1);
    dE_interaction *= -2 * si;
    dE_interaction_noise *= -2 * si;
    dE_norm *= -4 * si;
    dE_norm += 4 * N - 4;
}

void MF_Rad::MonteCarloStep(int i, int mu){
    double dE_interaction = 0.0;
    double dE_interaction_noise = 0.0;
    double dE_norm = 0.0;
    Delta_Energy(i, mu, dE_interaction, dE_interaction_noise, dE_norm);

    double dH = sqrt(lambda / N) * dE_interaction_noise + lambda / N * dE_interaction - 0.5 * lambda / N * dE_norm;
    double r = UNI_double(gen);

    if(r < exp(dH)){
        Flip_spin(state_X[i], mu);
        E_interaction += dE_interaction;
        E_interaction_noise += dE_interaction_noise;
        E_norm += dE_norm;
        H += dH;
    }
}
void MF_Rad::MonteCarloSweep(bool record_or_not){
    for(int i = 0; i < N; ++i) index_row[i] = N * UNI_double(gen);
    for(int i = 0; i < M; ++i) index_col[i] = M * UNI_double(gen);
    for(int row_idx = 0; row_idx < N; ++row_idx){
        for(int col_idx = 0; col_idx < M; ++col_idx){
            int i = index_row[row_idx];
            int mu = index_col[col_idx];
            MonteCarloStep(i, mu);
        }
    }
    if(start_record && record_or_not){
        history_X.push_back(state_X);
        history_interaction.push_back(E_interaction);
        history_norm.push_back(E_norm);
        history_H.push_back(H);
        history_interaction_noise.push_back(E_interaction_noise);
    }
}

void MF_Rad::Calculate_average_MMSE(){
    average_MMSE = 0.0;
    variance_MMSE = 0.0;
    double Var_ij = 0.0;
    double XiXj_av = 0.0;

    for(int i = 0; i < N; ++i){
        for(int j = 0; j < i; ++j){
            XiXj_av = 0;
            Var_ij = 0;
            //#pragma omp parallel for schedule(dynamic) reduction(+:XiXj_av)
            for(int t = 0; t < history_X.size(); ++t){
                XiXj_av += 1.0 * spin_inner_product(history_X[t][i], history_X[t][j], M);
            }
            XiXj_av /= 1.0 * history_X.size();
            average_MMSE += powf( XiXj_av - X0_inner_product[i][j], 2);

            //#pragma omp parallel for schedule(dynamic) reduction(+:Var_ij)
            for(int t = 0; t < history_X.size(); ++t){
                Var_ij += powf( spin_inner_product(history_X[t][i], history_X[t][j], M) - XiXj_av, 2);
            }
            variance_MMSE += Var_ij / history_X.size() * powf( X0_inner_product[i][j] - XiXj_av, 2.0);
        }
    }
    average_MMSE *= 2.0 / (1.0 * N*N*M);
    variance_MMSE *= 8.0 / (1.0 * M * N*N*M) / history_X.size();
}

void MF_Rad::Record_overlap(){
    std::vector<spin> X_T = transposeMatrix(state_X, N, M);
    std::vector<spin> X0_T = transposeMatrix(X0, N, M);
    std::vector<std::vector<int>> overlap(M, std::vector<int>(M, 0));
    for(int mu = 0; mu < M; ++mu){
        for(int nu = 0; nu < M; ++nu){
            overlap[mu][nu] = spin_inner_product(X_T[mu], X0_T[nu], N);
        }
    }
    history_overlap.push_back(overlap);
}


//Calculate the average overlap from history_X and X0
void MF_Rad::Calculate_average_overlap(){
    X0_overlap = Mat(M, std::vector<double>(M, 0.0));
    X0_overlap_sq = Mat(M, std::vector<double>(M, 0.0));
    average_overlap = Mat(M, std::vector<double>(M, 0.0));
    average_overlap_sq = Mat(M, std::vector<double>(M, 0.0));
    average_self_overlap = Mat(M, std::vector<double>(M, 0.0));
    average_self_overlap_sq = Mat(M, std::vector<double>(M, 0.0));
    sorted_overlap_sq = std::vector<double>(M * M, 0.0);
    std::vector<spin> X0_T = transposeMatrix(X0, N, M);
    int NMC = history_X.size();

    //Calculate the X0_overlap matrix
    for(int mu = 0; mu < M; ++mu){
        for(int nu = 0; nu <= mu; ++nu){
            double overlap_ = 1.0 * spin_inner_product(X0_T[mu], X0_T[nu], N) / N;
            X0_overlap[mu][nu] = overlap_;
            X0_overlap_sq[mu][nu] = overlap_ * overlap_;
            X0_overlap[nu][mu] = overlap_;
            X0_overlap_sq[nu][mu] = overlap_ * overlap_;
        }
    }

    //#pragma omp parallel for schedule(dynamic)
    for(int t = 0; t < NMC; ++t){
        std::set<int> not_used;
        for(int i = 0; i < M; ++i) not_used.insert(i);
        std::vector<int> Permutation(M);
        std::vector<int> parity(M);
        std::vector<spin> X_T = transposeMatrix(history_X[t], N, M);
        int max_overlap = 0;

        for(int mu = 0; mu < M ; ++mu){
            //Find the pattern nu that is closest to X0(mu) in absolute value, and record in Permutation[mu] = nu
            max_overlap = 0;
            for(const int &nu : not_used){
                int overlap = spin_inner_product(X0_T[mu], X_T[nu], N);
                if( abs(overlap) > max_overlap){
                    max_overlap = abs(overlap);
                    Permutation[mu] = nu;
                    parity[mu] = overlap > 0 ? 1 : -1;
                }
            }
            not_used.erase(Permutation[mu]);
        }
        std::vector<double> overlap_sq_vec(M*M, 0.0);
        for(int mu = 0; mu < M; ++mu){
            for(int nu = 0; nu < M; ++nu){
                //#pragma omp critical
                {
                double overlap_ = 1.0 * parity[Permutation[nu]] * spin_inner_product(X0_T[mu], X_T[Permutation[nu]], N) / N;
                double self_overlap_ = 1.0 * parity[Permutation[mu]] * parity[Permutation[nu]] * spin_inner_product(X_T[Permutation[mu]], X_T[Permutation[nu]], N) / N;
                average_overlap[mu][nu] += overlap_ / NMC;
                average_overlap_sq[mu][nu] += overlap_ * overlap_ / NMC;
                overlap_sq_vec[mu * M + nu] = overlap_ * overlap_;
                average_self_overlap[mu][nu] += self_overlap_ / NMC;
                average_self_overlap_sq[mu][nu] += self_overlap_ * self_overlap_ / NMC;
                }
            }
        }
        std::sort(overlap_sq_vec.begin(), overlap_sq_vec.end());
        //#pragma omp critical
        {
        for(int i = 0; i < M * M; ++i){
            sorted_overlap_sq[i] += overlap_sq_vec[i] / NMC;
        }
        }
    }
}




