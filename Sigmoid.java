/**
 * Author: Harrison Chen
 * Date of Creation: 3/12/24
 * 
 * Sigmoid represents a sigmoid activation function
 */
class Sigmoid extends ActivationFunction
{
/**
 * Sigmoid activation function
 */
   double F(double x) 
   {
      return 1.0 / (1.0 + Math.exp(-x));
   }

/**
 * Derivative of the sigmoid function
 */
   double deriv(double x) 
   {
      double fOfX = F(x);
      return fOfX * (1.0 - fOfX);
   }

/**
 * Returns Sigmoid as this is a Sigmoid activation function
 */
   String getName()
   {
      return "Sigmoid";
   }
} // class Sigmoid extends ActivationFunction
