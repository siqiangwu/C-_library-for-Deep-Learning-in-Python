#include "expression.h"

expression::expression(
    int expr_id,
    const char *op_name,
    const char *op_type,
    int *inputs,
    int num_inputs) :exprid(expr_id),opname(op_name),optype(op_type),input(inputs,inputs + num_inputs)  // member initializer of constructor 'expression class
{
}

void expression::add_op_param_double(
    const char *key,
    double value)
{
    op_param[key]=tensor(value);
}

void expression::add_op_param_ndarray(
    const char *key,
    int dim,
    size_t shape[],
    double data[])
{
    op_param[key]=tensor(dim, shape, data);
}

// Definition of getters

std::string expression::get_op_name() const {
    return opname;
}
std::string expression::get_op_type() const{
    return optype;
}
std::vector<int> expression::get_input() const{
    return input;
}
int expression::get_id() const {
    return exprid;
}

tensor expression::get_op_parameter(std::string value) const{
    return op_param.find(value)->second;
}