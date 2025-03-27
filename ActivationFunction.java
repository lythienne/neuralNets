/**
 * Author: Harrison Chen
 * Date of Creation: 3/12/24
 * 
 * Abstract class for activation functions allows the user to select the activation function they want to use without changing the code
 */

abstract class ActivationFunction 
{
/**
 * The activation function
 */
   abstract double F(double x);

/**
 * The derivative of the activation function
 */
   abstract double deriv(double x);

/**
 * Returns the name of this activation function
 */
   abstract String getName();
} // abstract class ActivationFunction 
