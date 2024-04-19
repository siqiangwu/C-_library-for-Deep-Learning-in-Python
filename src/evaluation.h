#ifndef EVALUATION_H
#define EVALUATION_H

#include "expression.h"
#include "tensor.h"

#include <iostream>
#include <stdexcept>
#include <algorithm>

class evaluation{

    std::vector<expression>expressions_;

public:

    evaluation(const std::vector<expression> &exprs);

    void add_kwargs_double(
        const char *key,
        double value);

    void add_kwargs_ndarray(
        const char *key,
        int dim,
        size_t shape[],
        double data[]);

    // return 0 for success
    int execute();

    // return the variable computed by the last expression
    tensor &get_result();

    // operators
    double Input(expression &expr_);
    double Add(expression &expr_);
    double Const(expression &expr_);
    double Mul(expression &expr_);
    double Sub(expression &expr_);
    double ReLU(expression &expr_);
    double Flatten(expression &expr_);
    double Input2d(expression &expr_);
    double Linear(expression &expr_);
    double MaxPool2d(expression &expr_);
    double Conv2d(expression &expr_);

private:
    double result_;
    std::vector<expression> exprs_vector_;          // store expressions
    std::map<std::string, tensor> kwargs_;          // store inputs
    std::map<int, tensor> vars_;                    // for evaluation 
    std::map<int, double> vals_;

}; // class evaluation

#endif // EVALUATION_H
