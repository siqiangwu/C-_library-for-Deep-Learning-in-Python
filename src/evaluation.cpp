#include <assert.h>
#include "evaluation.h"
//#include <cstring>
#include <iostream>
#include <algorithm>
#include "expression.h"


evaluation::evaluation(const std::vector<expression> &exprs)
    : result_(0), exprs_vector_(exprs)
{
}

void evaluation::add_kwargs_double(
    const char *key,
    double value)
{
    //printf("Value of input a: %f\n", value);
    //if(strcmp(key, "a") == 0) {
    kwargs_[key] = tensor(value);
    //}
}

void evaluation::add_kwargs_ndarray(
    const char *key,
    int dim,
    size_t shape[],
    double data[])
{
    kwargs_[key] = tensor(dim, shape, data);
}

int evaluation::execute()
{
    vars_.clear();
    double result_ = 0;
    // Going over all the expressions and executing evaluation over all
    for(auto &expr_: exprs_vector_){                                        // "auto" just in case of the different types
        std::string expr_op_name = expr_.get_op_name();
        std::string expr_op_type = expr_.get_op_type();
        std::cout << "Expression Operation Name" << expr_op_name << std::endl;
        std::cout << "Expression Operation Type" << expr_op_type << std::endl;

        //double x = 0, y = 0;
        // if (expr_op_type != "Input" && expr_op_type != "Const") {
        //    x = vars_[expr_.get_input()[0]];
        //    y = vars_[expr_.get_input()[1]];
        //}

        // Distinguishing operations
        if (expr_op_type == "Input") {
                result_ = Input(expr_); 
            }
        
        else if (expr_op_type == "Const") {
            result_ = Const(expr_);
            }
        else if (expr_op_type == "Add") {
                result_ = Add(expr_);
            }
        else if (expr_op_type == "Mul") {
                result_ = Mul(expr_);
            }
        else if (expr_op_type == "Sub") {
                result_ = Sub(expr_);
            }
        else if (expr_op_type == "ReLU") {
            result_ = ReLU(expr_);
            }
        else if (expr_op_type == "Flatten") {
                result_ = Flatten(expr_);
            }
        else if (expr_op_type == "Input2d") {
                result_ = Input2d(expr_);
            }
        else if (expr_op_type == "Linear") {
                result_ = Linear(expr_);
            }
        else if (expr_op_type == "MaxPool2d") {
                result_ = MaxPool2d(expr_);
            }
        else if (expr_op_type == "Conv2d") {
                result_ = Conv2d(expr_);
            }    


        std::cout << "Result from operation: " << result_ << std::endl;
    }  

    return 0;
}

tensor &evaluation::get_result()
{
    execute();
    return vars_[exprs_vector_.back().get_id()];vars_[exprs_vector_.back().get_id()];;
}

// Operators

double evaluation::Input(expression &expr_) {
    std::string expr_op_name = expr_.get_op_name();
    if (kwargs_.count(expr_op_name)>0) {
        tensor tensor_input = tensor(kwargs_[expr_op_name]);
        vars_[expr_.get_id()]=tensor_input;
        return *tensor_input.get_data_array();
    }
    throw std::invalid_argument("Argument not in map"); //with throw, 'return -1' is not needed
}

double addition(double x, double y) {
    return x + y;
}

double evaluation::Add(expression &expr_) {
    std::vector<int> inputs = expr_.get_input();
    tensor x = vars_[inputs[0]];
    tensor y = vars_[inputs[1]];
    assert(x.get_shape_array_reference() == y.get_shape_array_reference());  // assert to make sure
    
    std::vector<double> x_val_data = x.get_data_array_value();
    std::vector<double> y_val_data = y.get_data_array_value();

    std::vector<double> add_result(x_val_data.size());  // create a vector to store the result with same size 
    
    size_t const x_dim = x.get_dim(); 
    size_t const y_dim = y.get_dim();
    // check if they are scalars or not
    if (x_dim == 0 && y_dim == 0){
        double result{addition(x.item(), y.item())};
        vars_[expr_.get_id()] = tensor(result);
        return result;
    }
    // others: vectors and matrix
    // transform will be used to iterate throgh the beginning to the end of x's data while going from the beginning of y's data, it will add each element from a to the corresponding data in b abd store the result in add_result
    else { 
        std::transform(x_val_data.begin(), x_val_data.end(), 
                       y_val_data.begin(), add_result.begin(), addition);
        vars_[expr_.get_id()] = tensor(x.get_dim(), x.get_shape_array(), add_result.data());   // tensor with the resulting sum and store it vars_
        return add_result[0];
    }    
}
    //vector<double> x_value_data = x.get_data_array_value();
    //vector<double> y_value_data = y.get_data_array_value();
    //double sum[x_value_data.size()]; 
    //for (size_t i = 0; i > x_value_data.size(); ++i) {
    //        sum[i]
    //}


double evaluation::Const(expression &expr_){
    tensor const_ = expr_.get_op_parameter("value");
    vars_[expr_.get_id()] = const_;
    //tensor const_ = tensor(expr_.get_const("value"));
    //vars_[expr_.get_id()]=const_;
    return *const_.get_data_array();
}   


double Scalar_multiplication(double x, double y){ 
    return x*y;
}

double Vector_multiplication(vector<double> x, vector<double> y){
    assert(x.size() == y.size());  // if they are not the same dimension, operation is impossible to make
    double result = -1;
    for(size_t i = 0; i < x.size(); i++) result+= x[i]*y[i]; 
    return result;
}

double evaluation::Mul(expression &expr_) {
    std::vector<int> inputs = expr_.get_input();
    double mul_result = -1;  // global result is initialized, depending of the tensors, this result will have a different value when updated
    tensor x = vars_[inputs[0]];
    tensor y = vars_[inputs[1]];
    size_t const x_dim = x.get_dim(); 
    size_t const y_dim = y.get_dim();
    vector<size_t> const x_shape = x.get_shape_array_value();
    vector<size_t> const y_shape = y.get_shape_array_value();
    
    // Check if both of them are scalar, 1D (vector) or 2D tensor (matrix)
    assert(x_dim < 3 && y_dim < 3); 
    
    if (x_dim > 0 && y_dim > 0) {
        if (x_dim == 1 && y_dim == 1) {     // both 1d tensor (vector) --> result is a scalar (0d tensor)
            double result = Vector_multiplication(x.get_data_array_value(),y.get_data_array_value());
            mul_result = result;
            vars_[expr_.get_id()] = tensor(result);
        } 
        // Matrix operation: Dimensions must be mxn and kxp; 
        // where the second dimension of the first matrix and the first from the second must be the same --> n == k
        else {
            assert(x_shape[1] == y_shape[0]);   
            double result[x_shape[0] * y_shape[1]];  // Result initialization; the result will have the first dimension of the first matrix and the second of the second matrix
            for (size_t i = 0; i < x_shape[0]; i++) {
                for (size_t j = 0; j < y_shape[1]; j++) {
                    double cumulative = 0;
                    for (size_t k = 0; k < x_shape[1]; k++) {
                        cumulative += x.at(i,k)*y.at(k,j);
                    }
                    result[i*y_shape[1]+j] = cumulative;
                }
            }
            size_t shape[2] = {x_shape[0],y_shape[1]};
            mul_result = result[0];
            vars_[expr_.get_id()] = tensor(2, shape, result);
        }
    }
    // Vector or a matrix and a scalar multiplication
    else if (x_dim > 0 && y_dim == 0) {
        //std::cout << "First dimension of matrix " << x_shape[0] << std::endl; 
        vector<double> x_data = x.get_data_array_value();
        double result[x_data.size()];  // Result initialization; shape will be the same size as the matrix of vector
        for(size_t i = 0; i < x_data.size(); ++i) {
            result[i] = x_data[i]*y.item();
        } 
        size_t shape[x_shape.size()];
        for(size_t i = 0; i < x_shape.size(); ++i) {
            shape[i] = x_shape[i];
        }
        vars_[expr_.get_id()] = tensor(x.get_dim(),shape, result); 
        mul_result = *result;  //pointer
    }
    // Scalar and vector or matrix multiplication (similar to before, but in different order)
    else if (y_dim > 0 && x_dim == 0){
        vector<double> y_data = y.get_data_array_value();
        double result[y_data.size()]; // Result initialization; shape will be the same size as the matrix of vector
        for(size_t i = 0; i < y_data.size(); ++i) {
            result[i] = y_data[i]*x.item();
        } 
        size_t shape[y_shape.size()];
        for(size_t i = 0; i < y_shape.size(); ++i) {
            shape[i] = y_shape[i];
        }
        vars_[expr_.get_id()] = tensor(y.get_dim(),shape, result);
        mul_result = *result; //pointer
    }
    // Both tensors are scalars
    else { 
        double result{Scalar_multiplication(x.item(), y.item())};
        
        vars_[expr_.get_id()] = tensor(result);
        mul_result = result;
    }
    return mul_result;
}

// Substraction is the same a Addition but instead of "+" -> "-"

double substraction(double x, double y) {
    return x - y;
}

double evaluation::Sub(expression &expr_) {
    std::vector<int> inputs = expr_.get_input();
    tensor x = vars_[inputs[0]];
    tensor y = vars_[inputs[1]];
    assert(x.get_shape_array_reference() == y.get_shape_array_reference());  // assert to make sure
    
    std::vector<double> x_val_data = x.get_data_array_value();
    std::vector<double> y_val_data = y.get_data_array_value();

    std::vector<double> sub_result(x_val_data.size());  // create a vector to store the result with same size 

    size_t const x_dim = x.get_dim(); 
    size_t const y_dim = y.get_dim();

    // check if they are scalars or not
    if (x_dim == 0 && y_dim == 0){ 
        double result{substraction(x.item(), y.item())};
        vars_[expr_.get_id()] = tensor(result);
        return result;
    }
    // others: vectors and matrix
    // transform will be used to iterate throgh the beginning to the end of x's data while going from the beginning of y's data, it will substract each element from a to the corresponding data in b abd store the result in sub_result
    else { 
        std::transform(x_val_data.begin(), x_val_data.end(), 
                       y_val_data.begin(), sub_result.begin(), substraction);
        vars_[expr_.get_id()] = tensor(x.get_dim(), x.get_shape_array(), sub_result.data());   // tensor with the resulting sum and store it vars_
        return sub_result[0];
    }    
}

// Element wise ReLU operation
double evaluation::ReLU(expression &expr_) {
    tensor input = vars_[expr_.get_input()[0]];
    vector<double> input_data = input.get_data_array_value();
    vector<double> relu(input_data.size());
    std::transform(input_data.begin(), input_data.end(), relu.begin(),  // like in other functions, to avoid for loops transform is used
                   [](double x) { return std::max(0.0, x); });          //lambda function, 'x' is taken and the maximum value between 0.0 and 'x' is returned  --> ReLU function
    vars_[expr_.get_id()] = tensor(input.get_dim(), input.get_shape_array(), relu.data());
    return relu[0];
}

// Flatten: Each example is flattened into a vector using row-major order
double evaluation::Flatten(expression &expr_) {
    tensor input = vars_[expr_.get_input()[0]];
    vector<size_t> shape = input.get_shape_array_value();
    size_t N = shape[0];
    size_t CHW = shape[1] * shape[2] * shape[3];
    size_t flatten_shape[] = {N, CHW};
    vars_[expr_.get_id()] = tensor(2, flatten_shape, input.get_data_array()); 
    return float(N);
}

// Input2d: Obtain input tensor in NHWC format and output in NCHW format
double evaluation::Input2d(expression &expr_) {
    tensor input = kwargs_[expr_.get_op_name()];
    vector<size_t> shape = input.get_shape_array_value();
    size_t N = shape[0];
    size_t H = shape[1];
    size_t W = shape[2];
    size_t C = shape[3];
    size_t input2d_shape[] = {N,C,H,W};
    double change[N*H*W*C];
    tensor result(4, input2d_shape, change);
    for (size_t n = 0; n < N; n++){
        for (size_t h = 0; h < H; h++){
            for (size_t w = 0; w < W; w++){
                for (size_t c = 0; c < C; c++){
                    result.at(n,c,h,w) = input.at(n,h,w,c);                   
                }
            }
        }
    }
    vars_[expr_.get_id()] = result;
    return float(input2d_shape[0]);
}

tensor matrix_multiplication(tensor &x, tensor &y) {
    assert(x.get_shape_array_value()[1]==y.get_shape_array_value()[0]); // requirement for matrix multiplication
    size_t rows = x.get_shape_array_value()[0];
    size_t inner = x.get_shape_array_value()[1]; // columns of x
    size_t cols = y.get_shape_array_value()[1];
    vector<double> result(rows * cols, 0.0);    // Vector initialization of size rows & cols, which is the size of the resulting matrix

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            for (size_t k = 0; k < inner; k++) {
                result[i*cols + j] += x.at(i, k) * y.at(k, j);
            }
        }
    }
    size_t shape[]={rows, cols};
    return tensor(2, shape, result.data());
}

// Transpose of a matrix
void transpose(vector<double> &trans, size_t rows, size_t cols, tensor input)
{
    for (size_t i = 0; i < rows; i++) {
        for(size_t j = 0; j < cols; j++) {
            trans.push_back(input.at(j,i));  //append elements to vector
        }
    }
}

// Linear: for each row of x of the input tensor, a row in the output will be computed as (weight xT + bias)T
double evaluation::Linear(expression &expr_)
{
    tensor bias = expr_.get_op_parameter("bias");
    tensor weight = expr_.get_op_parameter("weight");
    std::vector<int> inputs = expr_.get_input();
    tensor input = vars_[inputs[0]];
    vector<size_t> input_shape = input.get_shape_array_value();
    vector<size_t> weight_shape = weight.get_shape_array_value();
    size_t batch_size = input_shape[0];
    size_t input_tensor_shape = input_shape[1];
    size_t output_tensor_shape = weight_shape[0];
    vector<double> input_T;                                     //transposed input initialization

    transpose(input_T, input_tensor_shape, batch_size, input);
    double *inputT = &input_T[0];                               //getting the pointer to the transposed input
    size_t shapeT[2] = {input_tensor_shape,batch_size};         //definition of the shape of the transposed input data
    tensor input_tensor_T = tensor(2, shapeT, inputT);          //definition of transposed input
    tensor result_mult = matrix_multiplication(weight, input_tensor_T);

    double *result_mult_pointer = result_mult.get_data_array();        // pointer to the data of the result
    // Adding the bias
    for (size_t i = 0; i < output_tensor_shape; i++) {
        for (size_t j = 0; j < batch_size; j++) {
            result_mult_pointer[i*batch_size + j] = result_mult_pointer[i*batch_size + j] + bias.get_data_array()[i];
        }
    }   
    vector<double> linear_result;
    // Another transpose to get the final result
    transpose(linear_result, batch_size, output_tensor_shape, result_mult);
    double *linear_result_pointer = &linear_result[0];
    size_t linear_result_shape[2] = {batch_size, output_tensor_shape};
    vars_[expr_.get_id()] = tensor(2, linear_result_shape, linear_result_pointer);
    return float(linear_result[0]);
}

// MaxPool2d: max pooling operation to the input tensor
double evaluation::MaxPool2d(expression &expr_) {
    tensor input = vars_[expr_.get_input()[0]];
    size_t kernel_size = static_cast<size_t>(expr_.get_op_parameter("kernel_size").item());  // convert kernel size to size_t using static_cast
    size_t N = input.get_shape_array_value()[0];
    size_t C = input.get_shape_array_value()[1];
    size_t H = input.get_shape_array_value()[2];
    size_t W = input.get_shape_array_value()[3];
   
    assert(H % kernel_size == 0 && W % kernel_size == 0);
    vector<double> result;
    for (size_t n = 0; n < N; n++) {                                                    // iterate over the batch
        for (size_t c = 0; c < C; c++) {                                                // iterate over the channels
            for (size_t h = 0; h < H; h += kernel_size) {                               // iterate over the height
                for (size_t w = 0; w < W; w += kernel_size) {                           // iterate over the width
                    double max_val = input.at(n, c, h, w);                              // max value initialization
                    for (size_t i = 0; i < kernel_size; i++) {                          // iterate over pooling window
                        for (size_t j = 0; j < kernel_size; j++) {      
                            max_val = std::max(max_val, input.at(n, c, h + i, w + j));  // max value of the pooling window
                        }
                    }
                    result.push_back(max_val);                                          // append max value
                }
            }
        }
    }
    size_t result_shape[4] = {N, C, H / kernel_size, W / kernel_size};                  // result shape tensor after pooling
    vars_[expr_.get_id()] = tensor(4, result_shape, result.data());                     // store output
    return result[0];
}

// Conv2d: 2D convolution between input and kernel
double evaluation::Conv2d(expression &expr_) {
    tensor input = vars_[expr_.get_input()[0]];
    tensor weight = expr_.get_op_parameter("weight");
    tensor bias = expr_.get_op_parameter("bias");

    // convert parameters to the appropiate type
    size_t out_channels = static_cast<size_t>(expr_.get_op_parameter("out_channels").item());
    size_t in_channels = static_cast<size_t>(expr_.get_op_parameter("in_channels").item());
    size_t kernel_size = static_cast<size_t>(expr_.get_op_parameter("kernel_size").item());
    size_t padding = static_cast<size_t>(expr_.get_op_parameter("padding").item());
    // get the dimension of the input tenspr
    size_t N = input.get_shape_array_value()[0];
    size_t H = input.get_shape_array_value()[2];
    size_t W = input.get_shape_array_value()[3];

    assert(padding == 0);  // making sure it is zero, function does not handle if it is not zero
    vector<double> result;
    // iterate over every element in the output tensor
    for (size_t n = 0; n < N; ++n) {
        for (size_t oc = 0; oc < out_channels; oc++) {
            for (size_t h = 0; h <= H - kernel_size; h++) {
                for (size_t w = 0; w <= W - kernel_size; w++) {
                    double convolution = 0.0;                                      // convolution sum initialization
                    for (size_t ic = 0; ic < in_channels; ic++) {
                        for (size_t kh = 0; kh < kernel_size; kh++) {
                            for (size_t kw = 0; kw < kernel_size; kw++) {
                                convolution += input.at(n, ic, h + kh, w + kw) *
                                            weight.at(oc, ic, kh, kw);
                            }
                        }
                    }
                    result.push_back(convolution + bias.at(oc));                    // store result
                }
            }
        }
    }

    size_t result_shape[4] = {N, out_channels, H - kernel_size + 1, W - kernel_size + 1};
    vars_[expr_.get_id()] = tensor(4, result_shape, result.data());
    return result[0];
}