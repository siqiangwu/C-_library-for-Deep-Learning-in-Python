#ifndef EXPRESSION_H
#define EXPRESSION_H

#include <vector>
#include <string>
#include <map>
#include <iostream>
#include "tensor.h"

class evaluation;

class expression
{
    friend class evaluation;

public:
    std::map<std::string,tensor>op_param;

    expression(
        int expr_id,
        const char *op_name,
        const char *op_type,
        int *inputs,
        int num_inputs);

    void add_op_param_double(
        const char *key,
        double value);

    void add_op_param_ndarray(
        const char *key,
        int dim,
        size_t shape[],
        double data[]);

    // Call getters
    std::string get_op_name() const;
    std::string get_op_type() const;
    std::vector<int> get_input() const;
    int get_id() const;
    tensor get_op_parameter(std::string value) const;

private:
    std::string opname;
    std::string optype;
    std::vector<int> input;
    int exprid;

}; // class expression

#endif // EXPRESSION_H
