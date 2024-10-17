// Code for calculating the MSE using Standard Metropolis-Hastings MCMC
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
    std::cout << "Usage: ./main -N <N> -M <M> -R <rep_arg> -n <NMC> -b <burnin> -p <flip_prob> -S <seed>" << std::endl;
    std::cout << "N            : > 0, number of rows." << std::endl;
    std::cout << "M            : > 0, number of columns." << std::endl;
    std::cout << "rep_arg      : file containing the list of lambdas." << std::endl;
    std::cout << "NMC          : number of Monte Carlo sweeps in total." << std::endl;
    std::cout << "burnin       : >= 0, out of NMC, number of Monte Carlo sweeps for burn-in." << std::endl;
    std::cout << "              for adaptive lambda, recommended to be larger than 10000." << std::endl;
    std::cout << "flip_prob    : probability of flipping the initial configuration from ground truth. " << std::endl;
    std::cout << "              for informative initialization, 0.0 . for random initialization, 0.5. " << std::endl;
    std::cout << "seed         : seed for true signal." << std::endl;
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
    int N = 0, M = 0, rep_num = 0, NMC = 0, NMC_burnin = -1, seed = 12345;
    double flip_prob = 0.5;
    std::string lambda_list;
    while((opt = getopt(argc, argv, "N:M:R:n:b:p:S:")) != -1){

        switch(opt){
            case 'N':
                check_int(optarg);
                N = std::atoi(optarg);
                break;
            case 'M':
                check_int(optarg);
                M = std::atoi(optarg);
                break;
            case 'R':
                lambda_list = optarg;
                break;
            case 'n':
                check_int(optarg);
                NMC = std::atoi(optarg);
                break;
            case 'b':
                check_int(optarg); 
                NMC_burnin = std::atoi(optarg);
                break;
            case 'p':
                check_float(optarg);
                flip_prob = std::atof(optarg);
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

    std::ofstream of( "Metropolis_data/N" + std::to_string(N) + "_M" + std::to_string(M) + "_seed" + std::to_string(seed) + "_NMC" + std::to_string(NMC) + "prob" + std::to_string(flip_prob) + ".txt" );

    std::vector<double> lambdas;
     std::ifstream ifs(lambda_list);
    double lambda;
    while (ifs >> lambda) lambdas.push_back(lambda);
    rep_num = lambdas.size();

    std::vector<double> current_exchange_rate(rep_num-1, 0.0);
    MultiCanonical mf(rep_num, N, M, lambdas, seed, seed + 1, flip_prob);
    std::cout << "Starting Monte Carlo simulation ...\n";
    #pragma omp parallel for schedule (dynamic)
    for(int i = 0; i < rep_num; ++i){
        for(int j = 0; j < NMC_burnin ; ++j) mf.replicas[i].MonteCarloSweep(false);
        mf.replicas[i].start_record = true;
        for(int j = NMC_burnin; j < NMC; ++j){
            mf.replicas[i].MonteCarloSweep(true);
        }
    }


    //Output the data
    std::cout << "Calculating physical quantities...\n";
    std::cout << "NOTE: This may take a while.\n";
    #pragma omp parallel for schedule (dynamic)
    for(int i = 0; i < rep_num; ++i) {
        mf.replicas[i].Calculate_average_MMSE();
    }

    std::cout << "Outputting data...\n";

    for(int i = 0; i < rep_num; ++i){
        of << std::setprecision(PRECISION) << lambdas[i] << " "; 
        of << mf.replicas[i].average_MMSE << " " << sqrt(mf.replicas[i].variance_MMSE) << " ";
        of << std::endl;
    }
    of.close();
    return 0;
}
