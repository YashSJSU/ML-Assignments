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
#include "/Users/yash/Desktop/HMM/Q2.11/createA.h"
using namespace std;


// for pi matrix
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

//for A, B matrix
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
    // a–z = 26
    {'a', 0}, {'b', 1}, {'c', 2}, {'d', 3}, {'e', 4}, {'f', 5},
    {'g', 6}, {'h', 7}, {'i', 8}, {'j', 9}, {'k', 10}, {'l', 11},
    {'m', 12}, {'n', 13}, {'o', 14}, {'p', 15}, {'q', 16}, {'r', 17},
    {'s', 18}, {'t', 19}, {'u', 20}, {'v', 21}, {'w', 22}, {'x', 23},
    {'y', 24}, {'z', 25},

    // punctuation = 27
    {'!', 26}, {'"', 27}, {'#', 28}, {'$', 29}, {'%', 30},
    {'&', 31}, {'\'', 32}, {'(', 33}, {')', 34}, {'*', 35},
    {'+', 36}, {',', 37}, {'-', 38}, {'.', 39}, {'/', 40},
    {':', 41}, {';', 42}, {'<', 43}, {'=', 44}, {'>', 45},
    {'?', 46}, {'@', 47}, {'[', 48}, {']', 49}, {'^', 50},
    {'_', 51}, {'`', 52}
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
    // a–z = 26
    {'a', 0}, {'b', 1}, {'c', 2}, {'d', 3}, {'e', 4}, {'f', 5},
    {'g', 6}, {'h', 7}, {'i', 8}, {'j', 9}, {'k', 10}, {'l', 11},
    {'m', 12}, {'n', 13}, {'o', 14}, {'p', 15}, {'q', 16}, {'r', 17},
    {'s', 18}, {'t', 19}, {'u', 20}, {'v', 21}, {'w', 22}, {'x', 23},
    {'y', 24}, {'z', 25},

    // punctuation = 27
    {'!', 26}, {'"', 27}, {'#', 28}, {'$', 29}, {'%', 30},
    {'&', 31}, {'\'', 32}, {'(', 33}, {')', 34}, {'*', 35},
    {'+', 36}, {',', 37}, {'-', 38}, {'.', 39}, {'/', 40},
    {':', 41}, {';', 42}, {'<', 43}, {'=', 44}, {'>', 45},
    {'?', 46}, {'@', 47}, {'[', 48}, {']', 49}, {'^', 50},
    {'_', 51}, {'`', 52}
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

void computeGammasAndDiGammas(vector<vector<double>>& alpha,vector<vector<double>>& beta,vector<vector<double>>& A,vector<vector<double>>& B,string obsSeq,vector<vector<double>>& gamma,vector<vector<vector<double>>>& diGamma,int T, int N)
{
    map<char, int> obsMap = {
    // a–z = 26
    {'a', 0}, {'b', 1}, {'c', 2}, {'d', 3}, {'e', 4}, {'f', 5},
    {'g', 6}, {'h', 7}, {'i', 8}, {'j', 9}, {'k', 10}, {'l', 11},
    {'m', 12}, {'n', 13}, {'o', 14}, {'p', 15}, {'q', 16}, {'r', 17},
    {'s', 18}, {'t', 19}, {'u', 20}, {'v', 21}, {'w', 22}, {'x', 23},
    {'y', 24}, {'z', 25},

    // punctuation = 27
    {'!', 26}, {'"', 27}, {'#', 28}, {'$', 29}, {'%', 30},
    {'&', 31}, {'\'', 32}, {'(', 33}, {')', 34}, {'*', 35},
    {'+', 36}, {',', 37}, {'-', 38}, {'.', 39}, {'/', 40},
    {':', 41}, {';', 42}, {'<', 43}, {'=', 44}, {'>', 45},
    {'?', 46}, {'@', 47}, {'[', 48}, {']', 49}, {'^', 50},
    {'_', 51}, {'`', 52}
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
    // a–z = 26
    {'a', 0}, {'b', 1}, {'c', 2}, {'d', 3}, {'e', 4}, {'f', 5},
    {'g', 6}, {'h', 7}, {'i', 8}, {'j', 9}, {'k', 10}, {'l', 11},
    {'m', 12}, {'n', 13}, {'o', 14}, {'p', 15}, {'q', 16}, {'r', 17},
    {'s', 18}, {'t', 19}, {'u', 20}, {'v', 21}, {'w', 22}, {'x', 23},
    {'y', 24}, {'z', 25},
    // punctuation = 27
    {'!', 26}, {'"', 27}, {'#', 28}, {'$', 29}, {'%', 30},
    {'&', 31}, {'\'', 32}, {'(', 33}, {')', 34}, {'*', 35},
    {'+', 36}, {',', 37}, {'-', 38}, {'.', 39}, {'/', 40},
    {':', 41}, {';', 42}, {'<', 43}, {'=', 44}, {'>', 45},
    {'?', 46}, {'@', 47}, {'[', 48}, {']', 49}, {'^', 50},
    {'_', 51}, {'`', 52}
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

}


//////////////////Function to print Pi (initial state probabilities)
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
    
    vector<char> observations = {
    'a','b','c','d','e','f','g','h','i','j','k','l','m',
    'n','o','p','q','r','s','t','u','v','w','x','y','z', // 0–25
    '!','"','#','$','%','&','\'','(',')','*','+',
    ',','-','.','/',';',':','<','=','>','?','@',
    '[',']','^','_','`' // 26–52
};
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


//////////////////////////////////////
vector<int> putative_plain_to_cipher(const vector<vector<double>>& B) {
    vector<int> key(26, 0);

    for (int state = 0; state < 26; ++state) {
        int bestObs = 0;
        double bestP = B[0][state];
        for (int obs = 1; obs < 26; ++obs) {
            if (B[obs][state] > bestP) { bestP = B[obs][state]; bestObs = obs; }
        }
        key[state] = bestObs; // plaintext state -> cipher letter index
    }
    return key;
}

double fraction_correct(const vector<int>& pred, const vector<int>& actual) {
    int correct = 0;
    for (int i = 0; i < 26; ++i) if (pred[i] == actual[i]) ++correct;
    return static_cast<double>(correct) / 26.0;
}

vector<int> bestStateForEachObservation(const vector<vector<double>>& B) {
    int S = (int)B.size();
    if (S == 0) return {};
    int M = (int)B[0].size();
    vector<int> argmax(M, -1);

    for (int k = 0; k < M; ++k) {
        int best = 0;
        double bestp = B[0][k];
        for (int i = 1; i < S; ++i) {
            if (B[i][k] > bestp) {
                bestp = B[i][k];
                best = i;
            }
        }
        argmax[k] = best;
    }
    return argmax;
}

/// Main Function ///////
int main(int argc, char *argv[])
{
    vector<char> observations = {
    'a','b','c','d','e','f','g','h','i','j','k','l','m',
    'n','o','p','q','r','s','t','u','v','w','x','y','z', // 0–25
    '!','"','#','$','%','&','\'','(',')','*','+',
    ',','-','.','/',';',':','<','=','>','?','@',
    '[',']','^','_','`' // 26–52
    };

    int z408[24][17] = {
        { 0, 1, 2, 3, 4, 3, 5, 6, 1, 7, 8, 9,10,11,12,10, 6},
        {13,14,15,16,17,18,19,20, 0,21, 2,22,23,24,25,18,16},
        {26,27,18,28, 5,29, 7,30,25,31,32,33,34,18,35,36,37},
        {38,39, 3, 0, 1, 6, 2, 8, 9,40, 5, 1,41, 9,42,25,43},
        { 7,28,44,26, 4,27,45,46,47,11,19,21,14,13,16,30,18},
        {22,15,25,17,35, 0,23,29,37,20,25,12,48,36,49,38,39},
        { 9,33,32,29,18,43,42, 8, 0,25,17, 6,31,20,38, 1, 6},
        {44,45, 3, 2, 1, 6,22,12,25,43,21,26, 5,28, 9, 9, 7},
        {50, 4,23,25,11,29,37,13,25,24,48,36,44,26,46, 0,51},
        { 6, 2,35, 9,15,27,10,20,47,33,39,16,43, 5,21, 7,19},
        { 4,50,11, 8,14,13,29,36,15,32,44,37,42,28, 9,20,21},
        {29, 0,35, 9,52,31,18,46,47,45,16, 3,22,12,27,34,40},
        { 2,36,26,48, 9, 5,32, 1,44,37,33,14,43,23,21,10,17},
        {46,29,24,27, 7,36, 0,48,44,26,42,33,40,37, 4,39, 2},
        {49, 5,11, 7,40, 0,51, 6,14,13,47,15,14,31,32, 8, 2},
        {28,10,38,46,42,41, 5,16,20,30,35,49,17, 1, 1,24,26},
        {33, 7,37,38,50,43, 3, 0, 1, 1, 4,41,40, 2,51, 6,14},
        {11,16,12,25,13,25,52,19,51,48,50,15,22, 0,40, 0, 6},
        { 1, 8,31,36, 9, 5,50,15,52,45,18,25,52,28,38,25,13},
        {14, 4,16,17,18,23,43,52,31,18,40, 0, 1,51,44,32,52},
        {21,24,19, 6,12, 0,49,12,40,35,45,47,30,44,24,10,25},
        {52,16,45,51,51,20,16,36, 2, 8, 9,12,34,19, 1,17,50},
        { 4,22,27,31,32,25,52,48,27,29,15,46, 6, 2,34,13,20},
        {14,43,12,46, 0,13,29,20,25,43,21,26,37,10,18,29, 7}
    };
    map<char, int> obsMap = {
    // a–z = 26
    {'a', 0}, {'b', 1}, {'c', 2}, {'d', 3}, {'e', 4}, {'f', 5},
    {'g', 6}, {'h', 7}, {'i', 8}, {'j', 9}, {'k', 10}, {'l', 11},
    {'m', 12}, {'n', 13}, {'o', 14}, {'p', 15}, {'q', 16}, {'r', 17},
    {'s', 18}, {'t', 19}, {'u', 20}, {'v', 21}, {'w', 22}, {'x', 23},
    {'y', 24}, {'z', 25},
    // punctuation = 27
    {'!', 26}, {'"', 27}, {'#', 28}, {'$', 29}, {'%', 30},
    {'&', 31}, {'\'', 32}, {'(', 33}, {')', 34}, {'*', 35},
    {'+', 36}, {',', 37}, {'-', 38}, {'.', 39}, {'/', 40},
    {':', 41}, {';', 42}, {'<', 43}, {'=', 44}, {'>', 45},
    {'?', 46}, {'@', 47}, {'[', 48}, {']', 49}, {'^', 50},
    {'_', 51}, {'`', 52}
    };
    map<int, char> revObsMap = {
        {0,'a'}, {1,'b'}, {2,'c'}, {3,'d'}, {4,'e'}, {5,'f'},
        {6,'g'}, {7,'h'}, {8,'i'}, {9,'j'}, {10,'k'}, {11,'l'},
        {12,'m'}, {13,'n'}, {14,'o'}, {15,'p'}, {16,'q'}, {17,'r'},
        {18,'s'}, {19,'t'}, {20,'u'}, {21,'v'}, {22,'w'}, {23,'x'},
        {24,'y'}, {25,'z'}, {26,'!'}, {27,'"'} ,{28,'#'}, {29,'$'}, {30,'%'},
        {31,'&'}, {32,'\''}, {33,'('}, {34,')'}, {35,'*'},
        {36,'+'}, {37,','}, {38,'-'}, {39,'.'}, {40,'/'},
        {41,':'}, {42,';'}, {43,'<'}, {44,'='}, {45,'>'},
        {46,'?'}, {47,'@'}, {48,'['}, {49,']'}, {50,'^'},
        {51,'_'}, {52,'`'}
    };

    if (argc < 3)
    {
        cerr << "Usage: " << argv[0] << "<Random Restarts> <min iterations>" << endl;
        return 1;
    }

    int N = 26;
    int M = 53;            // number of all possible observations ( distinct symbols in the zodiac text )
    int RR = stoi(argv[1]);
    int minIters = stoi(argv[2]);
    vector<double> RRstats(RR,0.0); // to store results of all the random restart runs

    cout << "Training HMM on zodiac text " << endl;
    cout << "Number of hidden states (N): " << N << endl;
    cout<<"Number of random restarts: "<< RR <<endl;

    //stays same 
    vector<vector<double>> A = createA("/Users/yash/Desktop/HMM/Q2.11/encryptedText.txt");
    //read the zodiac code from the matrix
    string message = "";
    for(int i=0; i<24; i++){
        for(int j=0; j<17; j++){
            char symbol = revObsMap.at(z408[i][j]);
            message += symbol;
        }
    }

    //random restarts and fixed iterations for reestimation
    int restarts = 0;
    while (restarts < RR) {
        cout<<"Starting Random Restart "<<restarts<<endl;
        vector<double> pi = generateUniformInitialProbabilities(N);
        vector<vector<double>> B = initialize_row_stochastic_matrix(N,M);
        // // Print initial parameters
        // cout << "INITIAL PARAMETERS:" << endl;
        // printHMMParameters(pi, A, B);
        for(int i=1; i<=minIters; i++){
            //restimate
            reEstimate(message, N, M, pi, A, B, i);
        }
        
        //get the putative key for all obs 
        vector<int> best = bestStateForEachObservation(B);
        // cout << "Best state per observation (symbol -> State):" << endl;
        // for (int k = 0; k < (int)best.size() && k < (int)observations.size(); ++k) {
        //     int i = best[k];
        //     // if (i >= 0) {
        //     //     cout << "  "<< observations[k]<< " -> "<< revObsMap[i]<<endl;
        //     //     }
        // }

        //check the accuracy
        double score = 0;
        for(int i=0; i<message.size(); i++){
            char sym = message[i];
            char pred = revObsMap[best[sym]];
            char truth = revObsMap[sym];
            cout<<truth<<"->"<<pred<<endl;
            if(truth == pred){
                score+=1;
            }
        }
        score = score/message.size();
        RRstats[restarts]=score;
        restarts++;
    }

    //print the accuracy after all the random restarts
    for(int i=0; i<RR; i++){
        cout<<"Restart "<<i+1<<" accuracy: "<<RRstats[i]<<endl;
    }
    return 0;
}