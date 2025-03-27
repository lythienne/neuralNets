/**
 * Author: Harrison Chen
 * Date of Creation: 1/24/24
 * 
 * The AB1Network class is an A-B-1 multilayer perceptron with gradient descent learning.
 */
class AB1Network
{
   final static int LAYERS = 2;                    //number of layers in the network
   final static int DERIVATIVE_OF_X = 1;           //derivative of F(x) = x with respect to x

   final static int AND = 0;                       //tests the network against AND
   final static int OR = 1;                        //tests the network against OR
   final static int XOR = 2;                       //tests the network against XOR

   final static int RANDOMIZE = 0;                 //randomly populates the weights
   final static int TWO_TWO_ONE = 1;               //populates the weights with preset values for a 2-2-1 network

   final static int[] PRESET_MODEL = {2, 2, 1};    //array constant that saves a 2-2-1 model to be checked against
   final static int a_LAYER_INDEX = 0;             //index of the number of input nodes in the preset model
   final static int h_LAYER_INDEX = 1;             //index of the number of hidden nodes in the preset model
   final static int F_LAYER_INDEX = 2;             //index of the number of final node in the preset model

   final static int NUM_TEST_CASES = 4;            //each test (AND, OR, XOR) has four test cases

   final static int a0INDEX = 0;                   //index of the first input value in the test cases array
   final static int a1INDEX = 1;                   //index of the second input value in the test cases array

   final static int MAX_WEIGHT_STRING_LENGTH = 20; //the maximum length the weight can be when printed so that they all line up



   int aNodes;                   //no, this is not a chemistry class unfortunately, a nodes is the number of nodes in the input layer
   int hNodes;                   //number of nodes in the hidden layer
   int FNodes;                   //number of nodes in the output layer (should be one)

   double[] a;                   //input activation layer
   double[] h;                   //hidden activation layer
   double[] F;                   //output activation layer (should only have one value)

   boolean train;                //whether to train (true) or run (false)
   int test;                     //AND, OR, or XOR

   double errorThreshold;        //maximum average error for training to end
   int maxIterations;            //maximum number of training cycles before training times out

   int weightPopulation;         //weight population method (randomized or 2-2-1 preset)
   double randomLow;             //randomized weight low bound (inclusive)
   double randomHigh;            //randomized weight high bound (exclusive)

   double lambda;                //learning factor

   double[][] wkj;               //2d weights array for the n=1 layer
   double[] wj0;                 //1d weights array for the n=2 layer (since theres only 1 output node)

   double[] theta_j;             //1d dot products array for the n=1 layer
   double theta_0;               //dot product for the n=2 layer

   double[] Omega_j;             //1d array for the uppercase Omega intermediate values for the n=1 layer
   double omega_0;               //lowercase omega value for the n=2 layer

   double[] Psi_j;               //1d array for the uppercase Psi intermediate values for the n=1 layer
   double psi_0;                 //lowercase psi value for the n=2 layer
   
   double[][] partialE_by_wkj;   //2d array of the partial derivatives of E with respect to the 1kj weights (in the n=1 layer)
   double[] partialE_by_wj0;     //1d array of the partial derivatives of E with respect to the 2j0 weights (in the n=2 layer)

   double[][] delta_wkj;         //2d array of the descents for each of the 1kj weights (in the n=1 layer)
   double[] delta_wj0;           //1d array of the descents for each of the 2j0 weights (in the n=2 layer)

   int iterations;               //number of iterations in the current training session
   double avgError;              //average error from the previous iteration

   double[][] testCases;         //2d array to store the two inputs for each of the four test cases
   double[] truth;               //1d array to store the truth values for each test case

   boolean printTruthTable;      //whether to print the truth table when reporting results
   boolean printWeights;         //whether to print the weights when reporting results

   double[] outputs;             //stores the output (F0) values from the final iteration for each test case to report to the user

/**
 * The user should set their values for the configuration parameters (that can be changed) here.
 */
   void setConfigurationParameters()
   {
      aNodes = 2;
      hNodes = 2;
      FNodes = 1;

      test = XOR;                     //tests are: AND, OR, XOR
      weightPopulation = RANDOMIZE;   //weight population methods are: RANDOMIZE, 2-2-1 

      train = true;                   //true = train network, false = run network
      
      lambda = 0.3;
      errorThreshold = 0.0002;
      maxIterations = 100000;
  
      randomLow = -1.5;
      randomHigh = 1.5;

      printTruthTable = true; 
      printWeights = false; 

   } // void setConfigurationParameters()

/**
 * Validates and echoes back all the configuration parameters to the user in an easily digestible format.
 * 
 * Throws ValidationErrors if the user selected nonsensical parameters or parameters that do not match the A-B-1 model.
 */
   void echoConfigurationParameters()
   {
      System.out.println("-----------------------------------------------------------");

      if (FNodes != PRESET_MODEL[F_LAYER_INDEX])   //check to see whether there is only one output node
         throw new ValidationError("Number of output nodes in an A-B-1 network should be 1, not "+FNodes);

      System.out.println("Configuration Parameters for a "+aNodes+"-"+hNodes+"-"+FNodes+" network with "+
            aNodes+" input nodes, "+hNodes+" hidden layer nodes, and "+ FNodes+" output node:\n");
      
      System.out.println("Weight Population Method:");
      switch (weightPopulation)
      {
         case RANDOMIZE:
            if(randomLow > randomHigh) 
               throw new ValidationError("Random low bound should be lower than the random high bound.");
            else
               System.out.println("Randomizing weights from "+randomLow+" to "+randomHigh);
            break;

         case TWO_TWO_ONE:
            if (aNodes == PRESET_MODEL[a_LAYER_INDEX])   //checks if the number of nodes agree with the 2-2-1 preset chosen
            {
               if (hNodes == PRESET_MODEL[h_LAYER_INDEX])
                  System.out.println("Using preset weights for a 2-2-1 network");
               else 
                  throw new ValidationError("Using 2-2-1 preset weights, number of hidden nodes should be 2, not "+hNodes);
            }
            else 
               throw new ValidationError("Using 2-2-1 preset weights, number of input nodes should be 2, not "+aNodes);
            break; // case TWO_TWO_ONE:

         default:
            throw new ValidationError("Please select a valid weight population method.");
      } // switch (weightPopulation)

      System.out.println();   //print another line for clarity

      String testName;        //convert integer to name to print
      switch (test)
      {
         case AND: testName = "AND"; break;
         case OR: testName = "OR"; break;
         case XOR: testName = "XOR"; break;
         default: throw new ValidationError("Please select a valid test (0 = AND, 1 = OR, 2 = XOR)");
      }

      if (train)
      {
         System.out.println("Network is training for "+testName+". \n\nRuntime Training Parameters:");
         System.out.println(" - Max Iterations: "+maxIterations);
         System.out.println(" - Error Threshold: "+errorThreshold);
         System.out.println(" - Lambda Value: "+lambda);
      }
      else
         System.out.println("Network is running against "+testName+".");

      System.out.println("-----------------------------------------------------------"); //print a separator for clarity
   } // void echoConfigurationParameters()

/**
 * Allocates memory for all major network arrays such as the activations, weights, test cases, and truth table.
 */
   void allocateMemory()
   {
      a = new double[aNodes];
      h = new double[hNodes];
      F = new double[FNodes];

      wkj = new double[aNodes][hNodes];
      wj0 = new double[hNodes];

      truth = new double[NUM_TEST_CASES];
      testCases = new double[NUM_TEST_CASES][aNodes];

      outputs = new double[NUM_TEST_CASES];

      if (train)
      {
         theta_j = new double[hNodes];

         Omega_j = new double[hNodes];
         Psi_j = new double[hNodes];

         partialE_by_wkj = new double[aNodes][hNodes];
         partialE_by_wj0 = new double[hNodes];

         delta_wkj = new double[aNodes][hNodes];
         delta_wj0 = new double[hNodes];

      } // if (train)
   } // void allocateMemory()

/**
 * Populates the weights arrays with the user defined population method and the other arrays with hardcoded values.
 * 
 * Arrays that were allocated space but are not populated here all contain 0.0 or the default primitive value java assigns.
 */
   void populateArrays()
   {
      if (weightPopulation == RANDOMIZE)
      {
         for (int k = 0; k < aNodes; k++)
            for (int j = 0; j < hNodes; j++)
               wkj[k][j] = randomize(randomLow, randomHigh);

         for (int j = 0; j < hNodes; j++)
            wj0[j] = randomize(randomLow, randomHigh);
      }
      else
      {
         populateWeights();
      }

      testCases[0][0] = 0.0;
      testCases[0][1] = 0.0;

      testCases[1][0] = 0.0;
      testCases[1][1] = 1.0;

      testCases[2][0] = 1.0;
      testCases[2][1] = 0.0;

      testCases[3][0] = 1.0;
      testCases[3][1] = 1.0;

      populateTruthTable();

   } // void populateArrays()

/**
 * Manually populates weights for a 2-2-1 network into the weights arrays
 */
   void populateWeights()
   {
      wkj[0][0] = 0.2;
      wkj[0][1] = 0.243;
      wkj[1][0] = 0.1;
      wkj[1][1] = 0.353;

      wj0[0] = 0.757;
      wj0[1] = 0.75;

   } // void populateWeights()

/**
 * Manually populates the truth table with the current test's (AND, OR, XOR) truth table
 */
   void populateTruthTable()
   {
      switch (test)
      {
         case AND:
            truth[0] = 0.0;
            truth[1] = 0.0;
            truth[2] = 0.0;
            truth[3] = 1.0;
            break;

         case OR:
            truth[0] = 0.0;
            truth[1] = 1.0;
            truth[2] = 1.0;
            truth[3] = 1.0;
            break;

         case XOR:
            truth[0] = 0.0;
            truth[1] = 1.0;
            truth[2] = 1.0;
            truth[3] = 0.0;
            break;

      } // switch (test)
   } // void populateTruthTable()

/**
 * Randomize returns a random double point floating number within the bounds [low, high)
 */
   double randomize(double low, double high)
   {
      return (high-low) * Math.random() + low;
   }

/**
 * Trains or runs the neural network based on the user's selected mode. Activations are calculated with a sigmoid
 * activation function, while training uses gradient descent with changes in weights all applied simultaneously.
 */
   void trainOrRun()
   {
      if (train)
      {
         iterations = 0;
         avgError = Double.MAX_VALUE;                                      //average error always starts above threshold

         while (iterations < maxIterations && avgError > errorThreshold)   //train until max iterations or error threshold reached
         {
            avgError = 0.0;                                                //reset accumulator for the error

            for (int testCase = 0; testCase < NUM_TEST_CASES; testCase++)  //iterate to minimize error in each test case
            {
               runTestCase(testCase);
               calculateGradientDescent(testCase);
               applyGradientDescent();
               runTestCase(testCase);
            }

            iterations++;
            avgError = avgError / (double) NUM_TEST_CASES;

         } // while (iterations < maxIterations && error < maxAvgError)

         for(int testCase = 0; testCase < NUM_TEST_CASES; testCase++)
            runTestCase(testCase);

      } // if (train)
      else
         for (int testCase = 0; testCase < NUM_TEST_CASES; testCase++)
            runTestCaseWithoutSaving(testCase);
   } // void trainOrRun()

/**
 * Runs the network for one test case, calculating the values of all the hidden nodes and the output node
 * does not save the dot products since it is not needed when only running
 */
   void runTestCaseWithoutSaving(int testCase)
   {
      populateInputNodes(testCase);
      
      double theta;

      for (int j = 0; j < hNodes; j++)              //iterate over the hidden nodes to calculate their values
      {
         theta = 0.0;

         for (int k = 0; k < aNodes; k++)           //iterate over the input nodes to calculate their dot product
            theta += a[k] * wkj[k][j];

         h[j] = activationFunction(theta);
      }

      theta = 0.0;

      for (int j = 0; j < hNodes; j++)              //iterate over the hidden nodes to calculate their dot product
         theta += h[j] * wj0[j];

      F[0] = activationFunction(theta);
      outputs[testCase] = F[0];                     //store output in outputs array to be later reported

   } // void runTestCaseWithoutSaving(int testCase)

/**
 * Runs the network for one test case, calculating the values of all the hidden nodes and the output node
 */
   void runTestCase(int testCase)
   {
      populateInputNodes(testCase);

      for (int j = 0; j < hNodes; j++)              //iterate over the hidden nodes to calculate their values
      {
         theta_j[j] = 0.0;

         for (int k = 0; k < aNodes; k++)           //iterate over the input nodes to calculate their dot product
            theta_j[j] += a[k] * wkj[k][j];

         h[j] = activationFunction(theta_j[j]);
      }

      theta_0 = 0.0;

      for (int j = 0; j < hNodes; j++)              //iterate over the hidden nodes to calculate their dot product
         theta_0 += h[j] * wj0[j];

      F[0] = activationFunction(theta_0);
      outputs[testCase] = F[0];                     //store output in outputs array to be later reported

   } // void runTestCase(int testCase)

/**
 * Populates the input nodes with the correct values for the given test case
 */
   void populateInputNodes(int testCase)
   {
      for (int k = 0; k < aNodes; k++)
         a[k] = testCases[testCase][k];
   }

/**
 * Calculates the delta weights for the given test case and stores them in the correct arrays
 */
   void calculateGradientDescent(int testCase)
   {
      omega_0 = truth[testCase] - F[0];
      psi_0 = omega_0 * derivative(theta_0);

      avgError += omega_0 * omega_0 / 2.0;

      for (int j = 0; j < hNodes; j++)       //iterate over hidden nodes to calculate j0 intermediates
      {
         Omega_j[j] = psi_0 * wj0[j];
         Psi_j[j] = Omega_j[j] * derivative(theta_j[j]);

         partialE_by_wj0[j] = -h[j] * psi_0;
         delta_wj0[j] = -lambda * partialE_by_wj0[j];

         for (int k = 0; k < aNodes; k++)    //iterate over input nodes as well to calculate kj intermediates
         {
            partialE_by_wkj[k][j] = -a[k] * Psi_j[j];
            delta_wkj[k][j] = -lambda * partialE_by_wkj[k][j];
         }
      } // for (int j = 0; j < hNodes; j++)
   } // void calculateGradientDescent(int testCase)

/**
 * Applies the delta weights by updating the weights arrays
 */
   void applyGradientDescent()
   {
      for (int j = 0; j < hNodes; j++)       //iterate over hidden nodes to update the j0 weights
      {
         wj0[j] += delta_wj0[j];

         for (int k = 0; k < aNodes; k++)    //iterate over input nodes to update the kj weights
            wkj[k][j] += delta_wkj[k][j];
      }
   } // void applyGradientDescent()

@Deprecated
/**
 * Linear activation function (f(x) = x)
 */
   double oldactivationFunction(double x)
   {
      return x;
   }

@Deprecated
/**
 * Derivative of the linear activation function
 */
   double oldderivative(double x)
   {
      return DERIVATIVE_OF_X;
   }

/**
 * Sigmoid activation function
 */
   double activationFunction(double x)
   {
      return 1.0 / (1.0 + Math.exp(-x));
   }

/**
 * Derivative of the sigmoid function
 */
   double derivative(double x)
   {
      double fOfX = activationFunction(x);
      return fOfX * (1.0 - fOfX);
   }

/**
 * Reports results back to the user in an easily digestible format
 */
   void reportResults()
   {
      System.out.println("Network ran on "+NUM_TEST_CASES+" test cases.\n"); 

      if (train)                              //print reason for training exit
      {
         System.out.print("Training exited because the");

         if (iterations >= maxIterations)
            System.out.print(" maximum iterations ("+maxIterations+
                  ") was reached");
         else if (iterations >= maxIterations && avgError <= errorThreshold)
            System.out.print(" and the");
         else if (avgError <= errorThreshold) 
            System.out.print(" error threshold ("+errorThreshold+") was reached");

         System.out.println(".");

         System.out.println(" - Iterations reached: "+iterations);
         System.out.println(" - Average error: "+avgError+"\n");
      }  // if (train)

      System.out.println("Results:\n");      //print the outputs (and truth table)
      if (printTruthTable)
      {
         System.out.println("a0 |a1 |Truth|Output");
         System.out.println("---|---|-----|-------------------");
         for (int testCase = 0; testCase < NUM_TEST_CASES; testCase++)
            System.out.println(testCases[testCase][0]+"|"+testCases[testCase][1]+"|"+truth[testCase]+"  |"+(outputs[testCase])+"");
      }
      else
      {
         System.out.println("a0 |a1 |Output");
         System.out.println("---|---|-------------------");
         for (int testCase = 0; testCase < NUM_TEST_CASES; testCase++)
            System.out.println(testCases[testCase][0]+"|"+testCases[testCase][1]+"|"+(outputs[testCase])+"");
      }

      if (printWeights)                         //print the weights (possibly)
      {
         System.out.println("\n\nWeights:");
         for (int j = 0; j < hNodes; j++)
         {
            for (int k = 0; k < aNodes; k++)
               System.out.print("\nw1"+k+j+": "+format(wkj[k][j]));
            
            System.out.print("  |w2"+j+"0: "+format(wj0[j]));
            System.out.print("\n----------------------------------------------------------");
         }
         System.out.println();   //print a newline for clarity
      } // if (printWeights)
   } // void reportResults()

/**
 * Returns a formatted, twenty one character long string for a weight so that all the weights are aligned when displaying them
 */
   String format(double weight)
   {
      String weightString = ""+Math.abs(weight);
      if (weightString.length() < MAX_WEIGHT_STRING_LENGTH)                  //if the weight is less than 20 characters long
      {
         int spaces = MAX_WEIGHT_STRING_LENGTH - weightString.length();

         for (int count = 0; count < spaces; count++)                        //pad the weight with spaces so they align
            weightString += " ";
      }
      if (weight > 0) return " "+weightString; else return "-"+weightString; //add a space or the negative sign back in

   } // String format(double weight)
} // class AB1Network