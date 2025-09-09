#pragma once
#include <vector>
#include <string>
using namespace std;

// Preferred: pass the file path in (no globals in headers)
vector<vector<double>> createA(const string& filepath);
