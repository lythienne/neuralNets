/**
 * Author: Harrison Chen
 * Date of Creation: 1/31/24
 * 
 * Validation error is an error thrown when encountering an invalid configuration parameter while echoing them.
 * An example would be setting the weights to be the preset for a 2-2-1 network while also asking for five input nodes.
 */
public class ValidationError extends RuntimeException
{
/**
 * The only constructor for Validation errors that should be used includes a reason for the error.
*/
    public ValidationError(String reason)
    {
        super(reason);
    }
}
