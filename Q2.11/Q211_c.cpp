#include <iostream>
#include <vector>
#include <fstream>
#include <unordered_map>
#include </Users/yash/Desktop/HMM/Q2.11/createA.h>
using namespace std;

vector<vector<double>> createA(const string& filepath){
    vector<vector<double>> A(26, vector<double>(26,0));
    unordered_map<char, int> m;
    m.reserve(26);
    for (char c = 'a'; c <= 'z'; ++c) 
        m.emplace(c, c - 'a');


    //read the file
    ifstream file(filepath);
    if(!file.is_open()){
        cerr<<"Error opening the dataset file";
    }

    string allText, line;
    while(getline(file,line) && allText.length()<=1000000 ){
        for( int i=15; i<line.length(); i++){

            char c = line[i];

            if (isalpha(c)) {
                allText.push_back(tolower(c));
            }
        }
    }

    //now we have the allText, parse it and fill the matrix
    for(int i=0; i<allText.length()-1; i++){
        char one = allText[i];
        char two = allText[i+1];

        int idx1 = m.at(one);
        int idx2 = m.at(two);

        A[idx1][idx2] += 1;
    }

    //add 5 to each element
    for(int i=0; i<26; i++){
        for(int j=0; j<26; j++){
            A[i][j]+=5;
        }
    }

    //normalize 
    for(int i=0; i<26; i++){
        int rowSum = 0;
        for(int j=0; j<26; j++){
            rowSum += A[i][j];
        }
        for(int j=0; j<26; j++){
            A[i][j] /= rowSum;
            cout<<A[i][j]<<" ";
        }
        cout<<endl;
    }
    return A;
}
