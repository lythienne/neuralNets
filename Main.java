/**
 * Author: Harrison Chen
 * Date of Creation: 1/24/24
 * 
 * The main class tests training and running the AB1 neural network.
 */
public class Main 
{
   /**
    * Main method creates an AB1Network, then executes each of its major methods in order to test the network.
    */
   public static void main(String[] args) 
   {
      AB1Network network = new AB1Network();
      network.setConfigurationParameters();
      network.echoConfigurationParameters();
      network.allocateMemory();
      network.populateArrays();
      network.trainOrRun();
      network.reportResults();
   }
}