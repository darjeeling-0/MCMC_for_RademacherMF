// Code for calculating the MSE, mutual information, and overlaps using replica exchange MCMC
// for the Matrix Factorization problem with Rademacher prior under a Bayes optimal setting. 
// by Koki Okajima, University of Tokyo (Oct. 2024).

#include "ReplicaExchange.hpp"
#include <iomanip>
#include "unistd.h"
#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60
#include <map>

//If using the adaptive SNR method, 
//the number of Monte Carlo steps to estimate the acceptance rate of replica exchange
//before adapting the sequence of lambdas.
static int ADAPT_FREQ = 200;
//Number of folds to use in the jackknife estimate
//to assess the mutual information. 
static int JACKKNIFE_FOLD = 20;
static int PRECISION = 9;

void printProgress(double percentage) {
    int val = (int) (percentage * 100);
    int lpad = (int) (percentage * PBWIDTH);
    int rpad = PBWIDTH - lpad;
    printf("\r%3d%% [%.*s%*s]", val, lpad, PBSTR, rpad, "");
}

void usage(){
    std::cout << "Usage: ./main -N <N> -M <M> -A -R <rep_arg> -n <NMC> -b <burnin> -s <MCsteps> -L <SNR_max> -S <seed>" << std::endl;
    std::cout << "N        : > 0, number of rows." << std::endl;
    std::cout << "M        : > 0, number of columns." << std::endl;
    std::cout << "A        : use adaptive lambda." << std::endl;
    std::cout << "rep_arg  : > if -A, then number of lambdas." << std::endl;
    std::cout << "         : otherwise, file containing the list of lambdas." << std::endl;
    std::cout << "NMC      : number of Monte Carlo steps in total." << std::endl;
    std::cout << "burnin   : >= 0, out of NMC, number of Monte Carlo steps for burn-in." << std::endl;
    std::cout << "           for adaptive lambda, recommended to be larger than 10000." << std::endl;
    std::cout << "MCsteps  : >0, number of Monte Carlo sweeps per exchange move." << std::endl;
    std::cout << "SNR_max  : maximum signal-to-noise ratio (> 0)." << std::endl;
    std::cout << "seed     : seed for true signal." << std::endl;
}

void check_int(std::string s){
    for(auto c : s){
        if(!isdigit(c)){
            std::cout << "Error: " << s << " is not a positive integer." << std::endl;
            exit(1);
        }
    }
}

void check_float(std::string s){
    bool dot = false;
    for(auto c : s){
        if(!isdigit(c) and c != '.'){
            std::cout << "Error: " << s << " is not a positive float." << std::endl;
            exit(1);
        }
        if(c == '.' and dot){
            std::cout << "Error: " << s << " is not a positive float." << std::endl;
            exit(1);
        }
        if(c == '.') dot = true;
    }
}



int main( int argc, char* argv[] ){
    int opt;
    int N = 0, M = 0, rep_num = 0, NMC = 0, NMC_burnin = -1, MCsteps = 10, seed = 12345;
    bool flag_ADAPT = false;
    std::string lambda_list;
    double SNR_max = 20.0;
    while((opt = getopt(argc, argv, "N:M:AR:n:b:s:L:S:")) != -1){

        switch(opt){
            case 'N':
                check_int(optarg);
                N = std::atoi(optarg);
                break;
            case 'M':
                check_int(optarg);
                M = std::atoi(optarg);
                break;
            case 'A':
                std::cout << flag_ADAPT << std::endl;
                flag_ADAPT = true;
                break;
            case 'R':
                if(flag_ADAPT == true){
                    check_int(optarg);
                    rep_num = std::atoi(optarg);
                }
                else lambda_list = optarg;
                break;
            case 'n':
                check_int(optarg);
                NMC = std::atoi(optarg);
                break;
            case 'b':
                check_int(optarg); 
                NMC_burnin = std::atoi(optarg);
                break;
            case 's':
                check_int(optarg);
                MCsteps = std::atoi(optarg);
                break;
            case 'L':
                check_float(optarg);
                SNR_max = std::atof(optarg);
                break;
            case 'S':
                check_int(optarg);
                seed = std::atoi(optarg);
                break;
            case '?':
                usage();
                return 1;
            default:
                usage();
                return 1;
        }
    }
    if (NMC_burnin < 0) NMC_burnin = NMC / 5;
    std::map<std::string, std::ofstream> ofs;
    ofs["main"] = std::ofstream( "MC_data/main/N" + std::to_string(N) + "_M" + std::to_string(M) + "_seed" + std::to_string(seed) + "_NMC" + std::to_string(NMC) + ".txt");
    ofs["overlap"] = std::ofstream( "MC_data/overlap/N" + std::to_string(N) + "_M" + std::to_string(M) + "_seed" + std::to_string(seed) + "_NMC" + std::to_string(NMC) + "_overlap.txt");
    ofs["overlap_sq"] = std::ofstream( "MC_data/overlap_sq/N" + std::to_string(N) + "_M" + std::to_string(M) + "_seed" + std::to_string(seed) + "_NMC" + std::to_string(NMC) + "_overlap_sq.txt");
    ofs["self_overlap"] = std::ofstream( "MC_data/self_overlap/N" + std::to_string(N) + "_M" + std::to_string(M) + "_seed" + std::to_string(seed) + "_NMC" + std::to_string(NMC) + "_self_overlap.txt");
    ofs["self_overlap_sq"] = std::ofstream( "MC_data/self_overlap_sq/N" + std::to_string(N) + "_M" + std::to_string(M) + "_seed" + std::to_string(seed) + "_NMC" + std::to_string(NMC) + "_self_overlap_sq.txt");
    
    std::vector<double> lambdas;
    if(!flag_ADAPT){
        std::ifstream ifs(lambda_list);
        double lambda;
        while (ifs >> lambda) lambdas.push_back(lambda);
        rep_num = lambdas.size();
    }
    else{
        lambdas = std::vector<double>(rep_num, 0.0);
        lambdas[0] = 0.0;
        lambdas[1] = .01;
        double lam_max = SNR_max;
        double pow =  exp( log(lam_max  / lambdas[1]) / (rep_num-2) );
        for(int i = 1; i < rep_num - 1; ++i) lambdas[i+1] = pow * lambdas[i];
    }
    std::vector<double> current_exchange_rate(rep_num-1, 0.0);
    MultiCanonical mf(rep_num, N, M, lambdas, seed, seed + 1, 0.0);
    std::cout << "Starting Monte Carlo simulation ...\n";
    std::cout << "Starting Burnin procedure...\n";
    for(int outer_loop = 1; outer_loop <= NMC_burnin; ++outer_loop){
        #pragma omp parallel for schedule (dynamic)
        for(int i = 0; i < rep_num; ++i){
            for(int j = 0; j < MCsteps; ++j) mf.replicas[i].MonteCarloSweep(false);
        }
        mf.Replica_Exchange(outer_loop % 2);
        if(outer_loop % ADAPT_FREQ == 0){
            for(int i = 0; i < rep_num-1; ++i){
                current_exchange_rate[i] = 1.0 * mf.replica_exchange_success[i] / mf.replica_exchange_attempt[i];
            }
            double acc_rate_max = *std::max_element(current_exchange_rate.begin(), current_exchange_rate.end());
            double acc_rate_min = *std::min_element(current_exchange_rate.begin(), current_exchange_rate.end());
            std::cout << "  Current exchange rate: ";
            std::cout << std::setprecision(3) << "Max: " << acc_rate_max << " Min: " << acc_rate_min << std::endl;
            if(flag_ADAPT){mf.Adapt_lambda();
                for(auto e : mf.lambdas) std::cout << e << " ";
                std::cout << std::endl;
                for(int r = 0; r < rep_num; ++r){
                    lambdas[r] = mf.lambdas[r];
                }
            }
        }
        printProgress(1.0 * outer_loop / NMC_burnin);
    }
    for(int i = 0; i < rep_num; ++i){
        mf.replicas[i].start_record = true;
    }
    std::cout << "\nBurnin procedure completed.\n";
    std::cout << "Starting main Monte Carlo procedure..." << std::endl;
    for(int outer_loop = NMC_burnin; outer_loop < NMC; ++outer_loop){
        #pragma omp parallel for schedule (dynamic)
        for(int i = 0; i < rep_num; ++i){
            for(int j = 0; j < MCsteps-1; ++j) mf.replicas[i].MonteCarloSweep(false);
            mf.replicas[i].MonteCarloSweep(true);
        }
        mf.Replica_Exchange(outer_loop % 2);
        if(outer_loop % 100 == 0){
            printProgress(1.0 * (outer_loop - NMC_burnin) /( NMC - NMC_burnin));
            std::flush(std::cout);
        }
    }

    //Output the data
    std::cout << "Calculating physical quantities...\n";
    std::cout << "NOTE: This may take a while.\n";
    #pragma omp parallel for schedule (dynamic)
    for(int i = 0; i < rep_num; ++i) {
        mf.replicas[i].Calculate_average_MMSE();
        mf.replicas[i].Calculate_average_overlap();
    }
    mf.Jackknife_MI_estimate( JACKKNIFE_FOLD );
    std::cout << "Outputting data...\n";
    std::cout << "NOTE: matrices are outputted in row-major order.\n";
    for(int i = 0; i < rep_num; ++i){
        for(auto &e : ofs){
            e.second << std::setprecision(PRECISION) << lambdas[i] << " "; 
        }
        double acc_rate = 1.0 * mf.replica_exchange_success[i] / mf.replica_exchange_attempt[i];
        if(i == rep_num - 1) acc_rate = -1;
        ofs["main"] << acc_rate << " " 
                    << mf.Jackknife_MI[i] << " " << sqrt(mf.Jackknife_MI_Variance[i]) << " " 
                    << mf.replicas[i].average_MMSE << " " << sqrt(mf.replicas[i].variance_MMSE);

        for(int mu = 0; mu < M; ++mu){
            for(int nu = 0; nu < M; ++nu){
                ofs["overlap"] << mf.replicas[i].average_overlap[mu][nu] << " ";
                ofs["overlap_sq"] << mf.replicas[i].average_overlap_sq[mu][nu] << " ";
                ofs["self_overlap"] << mf.replicas[i].average_self_overlap[mu][nu] << " ";
                ofs["self_overlap_sq"] << mf.replicas[i].average_self_overlap_sq[mu][nu] << " ";
            }
        }
        for(auto &e : ofs){
            e.second << std::endl;
        }
    }
    ofs["overlap"] << "Inf" << " ";
    ofs["overlap_sq"] << "Inf" << " ";
    ofs["self_overlap"] << "Inf" << " ";
    ofs["self_overlap_sq"] << "Inf" << " ";

    for(int mu = 0; mu < M; ++mu){
        for(int nu = 0; nu < M; ++nu){
            ofs["overlap"] << mf.replicas[0].X0_overlap[mu][nu] << " ";
            ofs["overlap_sq"] << mf.replicas[0].X0_overlap_sq[mu][nu] << " ";
            ofs["self_overlap"] << mf.replicas[0].X0_overlap[mu][nu] << " ";
            ofs["self_overlap_sq"] << mf.replicas[0].X0_overlap_sq[mu][nu] << " ";
        }
    }
    for(auto &e : ofs){
        e.second << std::endl;
        e.second.close();
    }
    return 0;
}
