#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <stdexcept>
#include <iostream>
#include <assert.h>

using namespace std;

class tensor {
public:
    tensor(); // initialization of tensor with 0 value
    explicit tensor(double v); // constructor of scalar; takes a double value and initializes a tensor as a scalar with that value
                               //'explicit' used to prevent compiler from using implicit conversions
    tensor(int dim, size_t shape[], double data[]); //constructor from C

    // Accessors
    int get_dim() const; // to get the dimension of the tensor (0 -> scalar; 1 -> vector; 2 -> matrix...)

    double item() const; // return a scalar value if tensor is a scalar
    double &item();      // return a reference to a scalar value of a tensor that can be modified

    double at(size_t i) const;
    double at(size_t i, size_t j) const;
    double at(size_t i, size_t j, size_t k) const;
    double at(size_t i, size_t j, size_t k, size_t l) const;
    double &at(size_t i, size_t j, size_t k, size_t l);  

    size_t *get_shape_array();
    double *get_data_array();
    double *get_data_array_back();
    vector<double> &get_data_array_reference();
    vector<size_t> &get_shape_array_reference();
    vector<double> get_data_array_value() const;
    vector<size_t> get_shape_array_value() const;
private:
    std::vector<size_t> shape_;  // shape of the tensor
    std::vector<double> data_;   // data contained in the tensor
}; // class tensor

#endif 