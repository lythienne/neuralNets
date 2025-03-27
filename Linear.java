/**
 * Author: Harrison Chen
 * Date of Creation: 3/12/24
 * 
 * Linear represents a linear activation function (f(x) = x)
 * 
 * Table of Contents:
 *    F(double);
 *    deriv(double);
 *    getName();
 */
class Linear extends ActivationFunction
{
   static final double DERIVATIVE_OF_X = 1;     //derivative of F(x) = x with respect to x

/**
 * Linear activation function (f(x) = x)
 */
   double F(double x) 
   {
      return x;
   }

/**
 * Derivative of the linear function
 */
   double deriv(double x) 
   {
      return DERIVATIVE_OF_X;
   }
   
/**
 * Returns linear as this is a linear activation function
 */
   String getName()
   {
      return "Linear";
   }
} // class Linear extends ActivationFunction
