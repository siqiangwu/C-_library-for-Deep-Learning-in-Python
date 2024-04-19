#include "tensor.h"

// Constructor implementation

// default constructor;  initialization of tensor with 0 value
tensor::tensor():
    data_(1,0) {       
    }

// one argument constructor; initialization of a scalar tensor
tensor::tensor(double v):
    data_(1,v) {
    }

// constructor from C; initialization of a tensor with dimension, shape and data as input
tensor::tensor(int dim, size_t shape[], double data[]):
    shape_(shape,shape+dim) {
        // calculate N as shape[0]*shape[1]*..*shape[dim-1]
        int N = shape[0];
        for (int i = 1; i<dim; i++){
            if (shape[i] == 0) continue;  // if shape 0 do not multiply
            N *= shape[i];
        }

        data_.assign(data, data+N); // initialization of data vector with values from data array
    }

// Accessors; assertions are used to guard againts misuse

// get number of dimensions of the tensor
int tensor::get_dim() const {
    return shape_.size();
}

// return scalar value of the tensor
double tensor::item() const {
    assert(shape_.empty()); // to ensure tensor is scalar
    return data_[0];
}

// return reference to the scalar value of the tensor AND can be modified
double &tensor::item() {
    assert(shape_.empty()); // to ensure tensor is scalar
    return data_[0]; 
}

// return value at a specified index in a vector (1D tensor)
double tensor::at(size_t i) const {
    assert(get_dim()==1);
    assert(i<shape_[0]);
    return data_[i];
}

// return value at a specified indez in a matrix (2D tensor)
double tensor::at(size_t i, size_t j) const {
    assert(get_dim()==2);
    assert((i<shape_[0])&&(j<shape_[1]));
    return data_[i*shape_[1]+j];
}

double tensor::at(size_t i, size_t j, size_t k) const
{
    assert(get_dim() == 3);
    assert((i < shape_[0]) && (j < shape_[1]) && (k < shape_[2]));
    return data_[i*shape_[1]*shape_[2]+j*shape_[2]+k];
}

double tensor::at(size_t i, size_t j, size_t k, size_t l) const
{
    assert(get_dim() == 4);
    assert((i < shape_[0]) && (j < shape_[1]) && (k < shape_[2]) && (l < shape_[3]));
    return data_[i*shape_[1]*shape_[2]*shape_[3]+j*shape_[2]*shape_[3]+k*shape_[3]+l];
}

double &tensor::at(size_t i, size_t j, size_t k, size_t l)
{
    assert(get_dim() == 4);
    assert((i < shape_[0]) && (j < shape_[1]) && (k < shape_[2]) && (l < shape_[3]));
    return data_[i*shape_[1]*shape_[2]*shape_[3]+j*shape_[2]*shape_[3]+k*shape_[3]+l];
}

size_t *tensor::get_shape_array() {
    return shape_.empty()?nullptr:&shape_[0];
}

double *tensor::get_data_array(){
    return &data_[0];
}

double *tensor::get_data_array_back(){
    return &data_.back();
}

vector<double> &tensor::get_data_array_reference() {
    return data_;
}

vector<size_t> &tensor::get_shape_array_reference() {
    return shape_;
}

vector<double> tensor::get_data_array_value() const {
    return data_;
}

vector<size_t> tensor::get_shape_array_value() const {
    return shape_;
}