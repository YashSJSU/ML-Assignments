#include <iostream>
#include <vector>
#include <string>
#include <map>
using namespace std;

//get the set of all possible observation sequences possible
void generateCombinations(string letters, string currentString, int length, vector<string>& possObs) {
    if (currentString.size() == length) {   
        possObs.push_back(currentString);
        return; 
    }
    for (char c : letters) {
        generateCombinations(letters, currentString + c, length, possObs); 
    }
}

//for every observation sequence calculate P(O/lambda)
float findProbability( string obsSeq, vector<string>& possStates, vector<vector<float>>& A, vector<vector<float>>& B, vector<float>& pi, map<char,int>& obsMap, map<char,int>& stateMap){
    char Ob_0 = obsSeq[0];
    char Ob_1 = obsSeq[1];
    char Ob_2 = obsSeq[2];
    char Ob_3 = obsSeq[3];

    float psum = 0;
    for( string state : possStates){

        char q0 = state[0];
        char q1 = state[1];
        char q2 = state[2];
        char q3 = state[3];

        float p  = pi[stateMap[q0]] * B[stateMap[q0]][obsMap[Ob_0]] * A[stateMap[q0]][stateMap[q1]] *
                   B[stateMap[q1]][obsMap[Ob_1]] * A[stateMap[q1]][stateMap[q2]] *
                   B[stateMap[q2]][obsMap[Ob_2]] * A[stateMap[q2]][stateMap[q3]] *
                   B[stateMap[q3]][obsMap[Ob_3]];

        psum = psum + p;
    }

    return psum;
}


int main() {
    string letters = "sml";
    string states = "HC";
    int length = 4;      
    map<char,int> stateMap = { {'H',0} , {'C',1} };
    map<char,int> obsMap = {{'s',0}, {'m',1}, {'l',2}};

    
    vector<vector<float>> A = {
        {0.7, 0.3},
        {0.4, 0.6}
    };
    vector<vector<float>> B = {
        {0.1, 0.4, 0.5},
        {0.7, 0.2, 0.1}
    };
    vector<float> pi = {0.6, 0.4};

    vector<string> possObs;
    vector<string> possStates;
    map<string,float> mp;

    //creating all possible combinations of observation sequences
    generateCombinations(letters, "", length, possObs);
    generateCombinations(states, "", length, possStates);

    //finding the probability for each sequence
    for( string obsSeq : possObs){
        float psum = findProbability( obsSeq, possStates, A, B, pi, obsMap, stateMap );
        if(mp.find(obsSeq)==mp.end()){
            mp[obsSeq] = psum;
        }
    }

    float totalSum = 0;
    cout<<"Observation Sequence"<<" "<<"psum"<<endl;
    for( auto item : mp){
        cout<<item.first<<" "<<item.second<<endl;
        totalSum = totalSum + item.second;
    }

    cout<<"The sum of all probabilities : "<<totalSum<<endl;

    return 0;
}
