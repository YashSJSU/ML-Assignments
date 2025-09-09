#include <iostream>
#include <fstream>
#include <string>
#include <random>
#include <numeric>
#include <map>
#include <cctype>
#include <iomanip>
#include <algorithm>  // fill
#include <cmath>      // log, fabs, isinf
#include <limits>     // numeric_limits
using namespace std;

vector<double> generateUniformInitialProbabilities(int N)
{
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dis(0.0, 1.0);

    vector<double> pi(N);
    for (int i = 0; i < N; ++i)
    {
        // Generate random values for each state
        pi[i] = dis(gen);
    }

    // Normalize so probabilities sum to 1
    double sum = accumulate(pi.begin(), pi.end(), 0.0);

    if (sum > 0)
    {
        // Avoid division by zero
        for (int i = 0; i < N; ++i)
        {
            pi[i] /= sum;
        }
    }
    else
    {
        // Fallback for an unlikely case, or use 1/n directly if preferred
        double uniform_prob = 1.0 / N;
        fill(pi.begin(), pi.end(), uniform_prob);
    }

    return pi;
}

vector<vector<double>> initialize_row_stochastic_matrix(int N, int M)
{
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dis(0.0, 1.0);

    vector<vector<double>> matrix(N, vector<double>(M));

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++)
        {
            matrix[i][j] = dis(gen);
        }

        double sum = accumulate(matrix[i].begin(), matrix[i].end(), 0.0);

        if (sum > 0)
        {
            // Avoid division by zero
            for (int j = 0; j < M; j++)
            {
                matrix[i][j] /= sum;
            }
        }
        else
        {
            // Fallback for an unlikely case, or use 1/n directly if preferred
            double uniform_prob = 1.0 / M;
            fill(matrix[i].begin(), matrix[i].end(), uniform_prob);
        }
    }
    return matrix;
}

//used to calculate the sum of prob of all possible state seq that give the obsSeq P(O/lambda)
vector<vector<double>> forward_pass(int N, string obsSeq, vector<double>& pi, vector<vector<double>>& A, vector<vector<double>>& B, vector<double>& scaler){
    int T = obsSeq.size();
    map<char, int> obsMap = {
        {'a', 0}, {'b', 1}, {'c', 2}, {'d', 3}, {'e', 4}, {'f', 5},
        {'g', 6}, {'h', 7}, {'i', 8}, {'j', 9}, {'k', 10}, {'l', 11},
        {'m', 12}, {'n', 13}, {'o', 14}, {'p', 15}, {'q', 16}, {'r', 17},
        {'s', 18}, {'t', 19}, {'u', 20}, {'v', 21}, {'w', 22}, {'x', 23},
        {'y', 24}, {'z', 25}
    };


    vector<vector<double>> alphas(T, vector<double>(N));

    //computer alpha_0(i) for all the states
    double c0 = 0;
    for( int i=0; i<N; i++){
        double a_0_i = pi[i]*B[i][obsMap[obsSeq[0]]];
        alphas[0][i] = a_0_i;
        c0 = c0 + a_0_i;
    }
    c0 = 1/c0;
    scaler[0]=c0;

    //scale the a_0_i
    for(int i=0; i<N; i++){
        alphas[0][i] = c0*alphas[0][i];
    }

    //calculate the alpha_t(i) for t = 1 to T-1
    for( int t=1 ; t<T; t++){
        double ct = 0;
        //for all states j at t-1 we transition to time t at state i
        for(int i=0; i<N; i++){
            double a_t_i = 0;
            for( int j=0; j<N; j++){
                a_t_i = a_t_i + alphas[t-1][j]*A[j][i];
            }
            //multiply by prob to see obs[t] at this state i at time t
            a_t_i = a_t_i * B[i][obsMap[obsSeq[t]]];
            alphas[t][i] = a_t_i;
            ct += a_t_i;
        }
        ct = 1/ct;
        scaler[t]=ct;

        //scale the a_t_i
        for(int i=0; i<N; i++){
            alphas[t][i] = ct*alphas[t][i];
        }
    }
    return alphas;
}

vector<vector<double>> backward_pass(int N, string obsSeq, vector<double>& pi, vector<vector<double>>& A, vector<vector<double>>& B, vector<double>& scaler){
    int T = obsSeq.size();
    map<char, int> obsMap = {
    {'a', 0}, {'b', 1}, {'c', 2}, {'d', 3}, {'e', 4}, {'f', 5},
    {'g', 6}, {'h', 7}, {'i', 8}, {'j', 9}, {'k', 10}, {'l', 11},
    {'m', 12}, {'n', 13}, {'o', 14}, {'p', 15}, {'q', 16}, {'r', 17},
    {'s', 18}, {'t', 19}, {'u', 20}, {'v', 21}, {'w', 22}, {'x', 23},
    {'y', 24}, {'z', 25}
    };

    vector<vector<double>> betas(T, vector<double>(N));

    // Initialize beta at final time step - should be scaled by c[T-1]
    for( int i=0; i<N; i++){
        betas[T-1][i] = 1.0;  // This part looks correct
    }

    //calculate the beta_t(i) for t = T-2 to 0
    for( int t=T-2; t>=0; --t ){
        for(int i=0; i<N; ++i){
            double b_t_i = 0.0;
            for( int j=0; j<N; ++j){
                b_t_i += A[i][j] * B[j][obsMap[obsSeq[t+1]]] * betas[t+1][j] ;
            }
            betas[t][i] = b_t_i * scaler[t];
        }
    }
    return betas;
}

void computeGammasAndDiGammas(
    vector<vector<double>>& alpha,
    vector<vector<double>>& beta,
    vector<vector<double>>& A,
    vector<vector<double>>& B,
    string obsSeq,
    vector<vector<double>>& gamma,
    vector<vector<vector<double>>>& diGamma,    
    int T, 
    int N)
{
    map<char, int> obsMap = {
        {'a',0},{'b',1},{'c',2},{'d',3},{'e',4},{'f',5},{'g',6},{'h',7},{'i',8},
        {'j',9},{'k',10},{'l',11},{'m',12},{'n',13},{'o',14},{'p',15},{'q',16},
        {'r',17},{'s',18},{'t',19},{'u',20},{'v',21},{'w',22},{'x',23},{'y',24},
        {'z',25}
    };

    gamma.assign(T, vector<double>(N, 0.0));
    diGamma.assign(max(0, T-1), vector<vector<double>>(N, vector<double>(N, 0.0)));

    // t = 0 .. T-2
    for (int t = 0; t <= T - 2; ++t) {
        double denom = 0.0;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                denom += alpha[t][i] * A[i][j] * B[j][obsMap[obsSeq[t + 1]]] * beta[t + 1][j];
            }
        }

        for (int i = 0; i < N; ++i) {
            double gamma_t_i = 0.0;
            for (int j = 0; j < N; ++j) {
                diGamma[t][i][j] =
                    (alpha[t][i] * A[i][j] * B[j][obsMap[obsSeq[t + 1]]] * beta[t + 1][j]) / denom;  // keeping your current formula
                gamma_t_i += diGamma[t][i][j];
            }
            gamma[t][i] = gamma_t_i; 
        }
    }

    // Special case: γ_{T-1}(i)
    double denom_last = 0.0;
    for (int i = 0; i < N; ++i) 
        denom_last += alpha[T - 1][i];
    for (int i = 0; i < N; ++i) 
        gamma[T - 1][i] = alpha[T - 1][i] / denom_last;
}




void reEstimate(string obsSeq, int N, int M, vector<double>& pi, vector<vector<double>>& A, vector<vector<double>>& B, int &iter){
    int T = obsSeq.size();
    map<char, int> obsMap = {
    {'a', 0}, {'b', 1}, {'c', 2}, {'d', 3}, {'e', 4}, {'f', 5},
    {'g', 6}, {'h', 7}, {'i', 8}, {'j', 9}, {'k', 10}, {'l', 11},
    {'m', 12}, {'n', 13}, {'o', 14}, {'p', 15}, {'q', 16}, {'r', 17},
    {'s', 18}, {'t', 19}, {'u', 20}, {'v', 21}, {'w', 22}, {'x', 23},
    {'y', 24}, {'z', 25}
    };

    vector<double> scaler(T,0);
    //do forward pass
    vector<vector<double>> alpha = forward_pass(N, obsSeq, pi, A ,B, scaler);
    cout<<"Completed forward pass for "<<iter<<endl;
    //do the backward pass
    vector<vector<double>> beta = backward_pass(N, obsSeq, pi, A, B, scaler);
    cout<<"Completed backward pass for "<<iter<<endl;
    //get gamma and di gammas
    vector<vector<double>> gamma;
    vector<vector<vector<double>>> di_gamma;
    computeGammasAndDiGammas(alpha, beta, A, B, obsSeq, gamma, di_gamma, T, N);
    cout<<"Calculated gamma and di-gamma for "<<iter<<endl;
    
    //reestimate pi
    for(int i=0; i<N; i++){
        pi[i] = gamma[0][i];
    }

    //re estimate A
    for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){
            double numer = 0;
            double denom = 0;
            for(int t=0; t<T-1; t++){
                numer += di_gamma[t][i][j];
                denom += gamma[t][i];
            }
            A[i][j] = numer/denom;
        }
    }

    //re estimate B
    for(int i=0; i<N; i++){
        for(int j=0; j<M; j++){
            double numer = 0;
            double denom = 0;
            for(int t=0; t<T; t++){
                if(obsMap[obsSeq[t]]==j){
                    numer += gamma[t][i];
                }
                denom += gamma[t][i];
            }
            B[i][j] = numer/denom;
        }
    }
    cout<<"Re-estimated the model for "<<iter<<'\n';

    //computer logP
    // double logP = 0;
    // for(int t=0; t<T; t++){
    //     logP += log(scaler[t]);
    // }
    // logP = -logP;
    // return logP;
}


// Function to print Pi (initial state probabilities)
void printPi(const vector<double>& pi) {
    cout << "Pi Matrix (Initial State Probabilities):" << endl;
    cout << "[ ";
    for (int i = 0; i < pi.size(); i++) {
        cout << fixed << setprecision(6) << pi[i];
        if (i < pi.size() - 1) cout << ", ";
    }
    cout << " ]" << endl << endl;
}
// Function to print A matrix (transition probabilities)
void printA(const vector<vector<double>>& A) {
    cout << "A Matrix (Transition Probabilities):" << endl;
    cout << "From\\To  ";
    for (int j = 0; j < A[0].size(); j++) {
        cout << "State" << j << "    ";
    }
    cout << endl;
    
    for (int i = 0; i < A.size(); i++) {
        cout << "State" << i << "   ";
        for (int j = 0; j < A[i].size(); j++) {
            cout << fixed << setprecision(6) << A[i][j] << "  ";
        }
        cout << endl;
    }
    cout << endl;
}
// Function to print B matrix (emission probabilities)
void printB(const vector<vector<double>>& B) {
    cout << "B Matrix (Emission Probabilities):" << endl;
    
    vector<char> observations = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'};
    
    // Print header with state labels
    cout << "obs\\state ";
    for (int j = 0; j < B.size(); j++) {
        cout << "State" << j << "   ";
    }
    cout << endl;
    
    // Print each observation row
    for (int i = 0; i < observations.size() && i < B[0].size(); i++) {
        cout << "    " << observations[i] << "     ";
        for (int j = 0; j < B.size(); j++) {
            cout << fixed << setprecision(6) << B[j][i] << " ";
        }
        cout << endl;
    }
    cout << endl;
}
// Function to print all matrices together
void printHMMParameters(const vector<double>& pi, 
                       const vector<vector<double>>& A, 
                       const vector<vector<double>>& B) {
    cout << "=== HMM PARAMETERS ===" << endl << endl;
    printPi(pi);
    printA(A);
    printB(B);
    cout << "======================" << endl << endl;
}



int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        cerr << "Usage: " << argv[0] << " <input_file> <N>" << endl;
        return 1;
    }

    string filename = argv[1];
    int N = stoi(argv[2]); // convert string to int
    int M = 26;            // number of all possible observations ( a to z and a space )

    cout << "Training HMM on the file: " << filename << endl;
    cout << "Number of hidden states (N): " << N << endl;

    vector<double> pi = generateUniformInitialProbabilities(N);
    vector<vector<double>> A = initialize_row_stochastic_matrix(N,N);
    vector<vector<double>> B = initialize_row_stochastic_matrix(N,M);

    // Print initial parameters
    cout << "INITIAL PARAMETERS:" << endl;
    printHMMParameters(pi, A, B);

    //read the lines form the file
    ifstream file(filename);
    if(!file.is_open()){
        cerr<<"Error opening the dataset file";
        return 1;
    }

    string allText, line;
    while(getline(file,line) && allText.length()<=50000 ){
        for( int i=15; i<line.length(); i++){
            char c = line[i];
            if (isalpha(c)) {
                allText.push_back(tolower(c));
            } else if (isspace(c)) {
                if (!allText.empty() && allText.back() != ' ')
                    allText.push_back(' ');
            }
        }
        if (!allText.empty() && allText.back() != ' ') allText.push_back(' ');
    }

    // cout<<allText<<endl;

    int minIters = 50;
    int iters = 0;

    while (iters < minIters) {
      reEstimate(allText, N, M, pi, A, B, iters);
      ++iters;
    }

    // Print final parameters after training
    cout << "FINAL PARAMETERS AFTER TRAINING:" << endl;
    printHMMParameters(pi, A, B);
    return 0;
}










