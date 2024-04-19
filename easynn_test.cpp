/**
 * A simple test program helps you to debug your easynn implementation.
 */

#include <stdio.h>
#include "src/libeasynn.h"

int main()
{
    program *prog = create_program();                          // a program is created by creating an instance

    int inputs0[] = {};                                        // array inputs0 is empty
    append_expression(prog, 0, "a", "Input", inputs0, 0);      // sets up an input expression "a" and is added to prog

    int inputs1[] = {0, 0};                                    // definition of an array with integer values 0 and 0
    append_expression(prog, 1, "", "Add", inputs1, 2);         // sets up expression that adds two values, takes the value from expression with ID '0' twice

    int inputs2[] = {0, 0};  
    append_expression(prog, 2, "", "Mul", inputs1, 2);

    int inputs3[] = {0, 0};  
    append_expression(prog, 3, "", "Sub", inputs1, 2);

    evaluation *eval = build(prog);                            // evaluation is built from the program
    add_kwargs_double(eval, "a", 5);                           // value 5 is assigned to the input value "a" 

    // execution of evaluation
    int dim = 0;                                
    size_t *shape = nullptr;
    double *data = nullptr;
    if (execute(eval, &dim, &shape, &data) != 0)               // executing the program
    {
        printf("evaluation fails\n");                          // if something fails --> print and exit the program (-1)
        return -1;
    }

    if (dim == 0)                                              // if the result is of dimension 0, the result is printing "Add" 
        printf("res = %f\n", data[0]);
    else
        printf("result as tensor is not supported yet\n");

    return 0;
}
