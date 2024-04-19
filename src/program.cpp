#include "program.h"
#include "evaluation.h"
#include "expression.h"

program::program()
{
}

void program::append_expression(
    int expr_id,
    const char *op_name,
    const char *op_type,
    int inputs[],
    int num_inputs)
{
    expression one_expression = expression(expr_id, op_name, op_type,inputs, num_inputs);
    expr_.push_back(one_expression);
}

int program::add_op_param_double(
    const char *key,
    double value)
{
    expr_.back().add_op_param_double(key,value);
    return 0;
}

int program::add_op_param_ndarray(
    const char *key,
    int dim,
    size_t shape[],
    double data[])
{
    expr_.back().add_op_param_ndarray(key,dim,shape,data);
    return 0;
}

evaluation *program::build()
{
    evaluation *eval_ = new evaluation(expr_);
    printf("evaluation %p\n", eval_);
    return eval_;
}
