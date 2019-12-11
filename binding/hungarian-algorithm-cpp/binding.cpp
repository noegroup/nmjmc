#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "Hungarian.h"

namespace py = pybind11;

vector<int> linear_sum_assignment(vector<vector<double>> cost_matrix) {
    HungarianAlgorithm HungAlgo;
    vector<int> assignment;
    double cost = HungAlgo.Solve(cost_matrix, assignment);
    return assignment;
}

PYBIND11_MODULE(hungarianMethod, m) {
    m.def("linear_sum_assignment", &linear_sum_assignment, "C++ implementation of the Hungarian method");
}