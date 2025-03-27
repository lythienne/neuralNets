import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Scanner;

/**
 * Author: Harrison Chen
 * Date of Creation: 1/24/24
 * 
 * The ABCNetwork class is an A-B-C multilayer perceptron with backpropagation.
 */
class ABCNetwork
{
   final static int LAYERS = 2;                       //number of layers in the network

   final static int AND = 0;                          //tests the network against AND
   final static int OR = 1;                           //tests the network against OR
   final static int XOR = 2;                          //tests the network against XOR

   final static int RANDOMIZE = 0;                    //randomly populates the weights
   final static int TWO_TWO_ONE = 1;                  //populates the weights with preset values for a 2-2-1 network
   final static int LOAD_FROM_FILE = 2;               //populates the weights from a file

   final static int[] PRESET_MODEL = {2, 2, 1};       //array constant that saves a 2-2-1 model to be checked against
   final static int a_LAYER_INDEX = 0;                //index of the number of input nodes in the preset model
   final static int h_LAYER_INDEX = 1;                //index of the number of hidden nodes in the preset model
   final static int F_LAYER_INDEX = 2;                //index of the number of final node in the preset model

   final static int NUM_TEST_CASES = 4;               //each test (AND, OR, XOR) has four test cases

   final static int a0INDEX = 0;                      //index of the first input value in the test cases array
   final static int a1INDEX = 1;                      //index of the second input value in the test cases array

   final static int MAX_WEIGHT_STRING_LENGTH = 24;    //the maximum length the weight can be when printed so that they all line up

   int aNodes;                   //no, this is not a chemistry class unfortunately, a nodes is the number of nodes in the input layer
   int hNodes;                   //number of nodes in the hidden layer
   int FNodes;                   //number of nodes in the output layer

   double[] a;                   //input activation layer
   double[] h;                   //hidden activation layer
   double[] F;                   //output activation layer

   boolean train;                //whether to train (true) or run (false)

   double errorThreshold;        //maximum average error for training to end
   int maxIterations;            //maximum number of training cycles before training times out
   double time;                  //stores the time taken to complete running/training

   int weightPopulation;         //weight population method (randomized, 2-2-1 preset, or loaded from file)
   double randomLow;             //randomized weight low bound (inclusive)
   double randomHigh;            //randomized weight high bound (exclusive)

   double lambda;                //learning factor
   ActivationFunction f;         //the activation function used to calculate the values of the activations

   double[][] wkj;               //2d weights array for the n=1 layer
   double[][] wji;               //2d weights array for the n=2 layer

   double[] theta_j;             //1d dot products array for the n=1 layer
   double[] psi_i;               //1d array for the lowercase psi intermediate values for the n=2 layer

   int iterations;               //number of iterations in the current training session
   double avgError;              //average error from the previous iteration

   double[][] testCases;         //2d array to store the inputs for each test case
   double[][] truth;             //2d array to store the truth values for each test case

   boolean printTruthTable;      //whether to print the truth table when reporting results
   boolean printWeights;         //whether to print the weights when reporting results

   double[][] outputs;           //stores the output (F0) values from the final iteration for each test case to report to the user

   static boolean saveWeights;   //whether to save the weights
   static String origin;         //used by the error handler to save which major method threw the error

   String saveWeightFile;        //the file to where to save the weights 
   String loadWeightFile;        //the file from where to load the weights
   String truthTableFile;        //the file where the truth table is stored
   String inputsFile;             //the file where the inputs are stored

/**
 * The user should set their values for the configuration parameters (that can be changed) here.
 */
   void setConfigurationParameters(String configFile)
   {
      origin = "setting the configuration parameters from file '"+configFile+"'";

      try
      {
         Scanner in = new Scanner(new File(configFile));

         while (in.hasNextLine())
         {
            String line = in.nextLine();

            while (!Parser.setString(line))  //while there is no value in the line, skip the line
               line = in.nextLine();

            line = Parser.cleanString(line);
            parseLine(line);
         } // while (in.hasNextLine()) 

         in.close();
      } // try
      catch (FileNotFoundException e)
      {
         throw new ValidationError("File \""+configFile+"\" does not exist.");
      } // try catch (FileNotFoundException e)
   } // void setConfigurationParameters()

/**
 * Parses a line in the config file to assign the variable in the line to the value in the line
 */
   void parseLine(String line)
   {
      switch (line)
      {
         case "aNodes": aNodes = Parser.extractInt(); break;
         case "hNodes": hNodes = Parser.extractInt(); break;
         case "FNodes": FNodes = Parser.extractInt(); break;

         case "weightPopulation": 
            String weightPopulationString = Parser.extractString().toUpperCase();

            switch (weightPopulationString)
            {
               case "RANDOMIZE": weightPopulation = RANDOMIZE; break;
               case "2-2-1": weightPopulation = TWO_TWO_ONE; break;
               case "LOAD FROM FILE": weightPopulation = LOAD_FROM_FILE; break;

               default: throw new ValidationError(weightPopulationString+" is not a valid weight population method.");
            } // switch (weightPopulationString)
            break; // case "weightPopulation":

         case "train": train = Parser.extractBoolean(); break;
         case "saveWeights": saveWeights = Parser.extractBoolean(); break;

         case "printTruthTable": printTruthTable = Parser.extractBoolean(); break;
         case "printWeights": printWeights = Parser.extractBoolean(); break;

         case "f", "activationFunction": 
            String activationFunctionString = Parser.extractString().toUpperCase();

            switch (activationFunctionString)
            {
               case "SIGMOID": f = new Sigmoid(); break;
               case "LINEAR": f = new Linear(); break;

               default: throw new ValidationError(activationFunctionString+" is not a valid activation function.");
            }
            break; // case "f", "activationFunction":

         case "lambda": lambda = Parser.extractDouble(); break;
         case "maxIterations": maxIterations = Parser.extractInt(); break;
         case "errorThreshold": errorThreshold = Parser.extractDouble(); break;

         case "randomLow": randomLow = Parser.extractDouble(); break;
         case "randomHigh": randomHigh = Parser.extractDouble(); break;

         case "saveWeightFile": saveWeightFile = Parser.extractString(); break;
         case "loadWeightFile": loadWeightFile = Parser.extractString(); break;
         case "truthTableFile": truthTableFile = Parser.extractString(); break;
         case "inputsFile": inputsFile = Parser.extractString(); break;

      } // switch (line)
   } // void assignValue(String line)

/**
 * Validates and echoes back all the configuration parameters to the user in an easily digestible format.
 * 
 * Throws ValidationErrors if the user selected nonsensical parameters or parameters that do not match the A-B-1 model.
 */
   void echoConfigurationParameters()
   {
      origin = "echoing and validating the configuration parameters";

      System.out.println("-----------------------------------------------------------");

      System.out.println("Configuration Parameters for a "+aNodes+"-"+hNodes+"-"+FNodes+" network with "+
            aNodes+" input nodes, "+hNodes+" hidden layer node(s), and "+ FNodes+" output node(s):\n");
      
      System.out.println("Weight Population Method:");
      switch (weightPopulation)
      {
         case RANDOMIZE:
            if (randomLow > randomHigh) 
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

         case LOAD_FROM_FILE:
            try
            {
               if (loadWeightFile == null)
                  throw new ValidationError("Load weight file not declared in the config file.");

               echoValidateLoadWeights();
            }
            catch (FileNotFoundException fe)
            {
               throw new ValidationError("Weights file '"+loadWeightFile+"' does not exist.");
            } //try catch (FileNotFoundException fe)
            catch (IOException ioe)
            {
               throw new ValidationError("Stream closed when reading from weights file or other IO Exception caught.");
            } //try catch (IOException ioe)
            break; // case LOAD_FROM_FILE:

         default:
            throw new ValidationError("Please select a valid weight population method.");
      } // switch (weightPopulation)

      System.out.println();   //print another line for clarity

      if (truthTableFile == null)
         throw new ValidationError("Truth table file not declared in the config file.");
      else
         System.out.println("Using the truth table stored in '"+truthTableFile+"'");

      if (inputsFile == null)
         throw new ValidationError("Inputs file not declared in the config file.");
      else
         System.out.println("Using the inputs stored in '"+inputsFile+"'");

      if (saveWeights)
      {
         if (saveWeightFile == null)
            throw new ValidationError("Save weight file not declared in the config file.");

         System.out.println("Saving the weights to '"+saveWeightFile+"'");
      }

      System.out.println();   //print another line for clarity

      if (train)
      {
         System.out.println("Network is training \n\nRuntime Training Parameters:");
         System.out.println(" - Activation Function: "+f.getName());
         System.out.println(" - Max Iterations: "+maxIterations);
         System.out.println(" - Error Threshold: "+errorThreshold);
         System.out.println(" - Lambda Value: "+lambda);
      }
      else
         System.out.println("Network is running");

      System.out.println("-----------------------------------------------------------"); //print a separator for clarity
   } // void echoConfigurationParameters()

/**
 * Echoes and validates the weights file if the user chooses to load the weights from a file
 */
   void echoValidateLoadWeights() throws FileNotFoundException, IOException
   {
      File file = new File(loadWeightFile);
      if (!file.isFile())
         throw new ValidationError("Weights file '"+loadWeightFile+"' does not exist.");

      DataInputStream in = new DataInputStream(new FileInputStream(file));

      int file_aNodes = (int) in.readDouble();     //reads the connectivity pattern from the saved weights file
      int file_hNodes = (int) in.readDouble();
      int file_FNodes = (int) in.readDouble();

      in.close();

      if (file_aNodes != aNodes || file_hNodes != hNodes || file_FNodes != FNodes)
      {
         throw new ValidationError("Saved connectivity pattern '"+file_aNodes+"-"+file_hNodes+"-"+file_FNodes+
               "' in the weights file '"+loadWeightFile+"' does not match user set '"+aNodes+"-"+hNodes+"-"+FNodes+"' in config file");
      }

      System.out.println("Loading '"+file_aNodes+"-"+file_hNodes+"-"+file_FNodes+"' weights from '"+loadWeightFile+"'");
   } // void echoValidateLoadWeights() throws FileNotFoundException, IOException

/**
 * Allocates memory for all major network arrays such as the activations, weights, test cases, and truth table.
 */
   void allocateMemory()
   {
      origin = "allocating memory for the arrays";

      a = new double[aNodes];
      h = new double[hNodes];
      F = new double[FNodes];

      wkj = new double[aNodes][hNodes];
      wji = new double[hNodes][FNodes];

      truth = new double[NUM_TEST_CASES][FNodes];
      testCases = new double[NUM_TEST_CASES][aNodes];

      outputs = new double[NUM_TEST_CASES][FNodes];

      if (train)
      {
         theta_j = new double[hNodes];
         psi_i = new double[FNodes];
      }
   } // void allocateMemory()

/**
 * Populates the weights arrays with the user defined population method and the other arrays with hardcoded values.
 * 
 * Arrays that were allocated space but are not populated here all contain 0.0 or the default primitive value java assigns.
 */
   void populateArrays()
   {
      origin = "populating the arrays";

      switch (weightPopulation)
      {
         case RANDOMIZE:
            for (int k = 0; k < aNodes; k++)
               for (int j = 0; j < hNodes; j++)
                  wkj[k][j] = randomize(randomLow, randomHigh);

            for (int j = 0; j < hNodes; j++)
               for (int i = 0; i < FNodes; i++)
                  wji[j][i] = randomize(randomLow, randomHigh);
            break; // case RANDOMIZE:
            
         case LOAD_FROM_FILE: loadWeights(); break;

         default: populateWeightsManually(); break;
      } // switch (weightPopulation)

      populateTestCases();
      populateTruthTable();

   } // void populateArrays()

/**
 * Manually populates weights for a 2-2-1 network into the weights arrays
 */
   void populateWeightsManually()
   {
      wkj[0][0] = 0.2;
      wkj[0][1] = 0.243;
      wkj[1][0] = 0.1;
      wkj[1][1] = 0.353;

      wji[0][0] = 0.75;
      wji[1][0] = 0.757;

   } // void populateWeightsManually()

/**
 * Saves the weights in binary to a file
 */
   void saveWeights()
   {
      try
      {
         File file = new File(saveWeightFile);
         DataOutputStream out = new DataOutputStream(new FileOutputStream(file));

         out.writeDouble(aNodes);
         out.writeDouble(hNodes);
         out.writeDouble(FNodes);

         for (int k = 0; k < aNodes; k++)
            for (int j = 0; j < hNodes; j++)
               out.writeDouble(wkj[k][j]);

         for (int j = 0; j < hNodes; j++)
            for (int i = 0; i < FNodes; i++)
               out.writeDouble(wji[j][i]);
         
         out.close();
      } // try
      catch (FileNotFoundException fe)
      {
         throw new ValidationError("Weights file '"+loadWeightFile+"' cannot be created or opened.");
      } // try catch (FileNotFoundException fe)
      catch (IOException ioe)
      {
         throw new ValidationError("Stream closed when saving to weights file or other IO Exception caught.");
      } // try catch (IOException ioe)
   } // void saveWeights()

/**
 * Loads the weights in binary from a file
 */
   void loadWeights()
   {
      try
      {
         File file = new File(loadWeightFile);
         DataInputStream in = new DataInputStream(new FileInputStream(file));

         in.readDouble();     //skips three doubles which are the network connectivity model
         in.readDouble();
         in.readDouble();

         for (int k = 0; k < aNodes; k++)
            for (int j = 0; j < hNodes; j++)
               wkj[k][j] = in.readDouble();

         for (int j = 0; j < hNodes; j++)
            for (int i = 0; i < FNodes; i++)
               wji[j][i] = in.readDouble();
         
         in.close();
      } // try
      catch (FileNotFoundException fe)
      {
         throw new ValidationError("Weights file '"+loadWeightFile+"' does not exist.");
      } //try catch (FileNotFoundException fe)
      catch (IOException ioe)
      {
         throw new ValidationError("Stream closed when loading from weights file or other IO Exception caught.");
      } //try catch (IOException ioe)
   } // void loadWeights()

@Deprecated
/**
 * Manually populates the truth table with the current test's (AND, OR, XOR) truth table
 */
   void AB1populateTruthTable()
   {
      truth[0][0] = 0.0;
      truth[1][0] = 1.0;
      truth[2][0] = 1.0;
      truth[3][0] = 0.0;

   } // void oldpopulateTruthTable()

@Deprecated
/**
 * Manually populates the truth table with the current tests' (AND, OR, XOR) truth tables
 */
   void ABCpopulateTruthTable()
   {
      truth[0][AND] = 0.0;
      truth[1][AND] = 0.0;
      truth[2][AND] = 0.0;
      truth[3][AND] = 1.0;
      
      truth[0][OR] = 0.0;
      truth[1][OR] = 1.0;
      truth[2][OR] = 1.0;
      truth[3][OR] = 1.0;

      truth[0][XOR] = 0.0;
      truth[1][XOR] = 1.0;
      truth[2][XOR] = 1.0;
      truth[3][XOR] = 0.0;
   } // void populateTruthTable()

@Deprecated
/**
 * Manually populates the test cases array with all the test cases
 */
   void ABCpopulateTestCases()
   {
      testCases[0][0] = 0.0;
      testCases[0][1] = 0.0;

      testCases[1][0] = 0.0;
      testCases[1][1] = 1.0;

      testCases[2][0] = 1.0;
      testCases[2][1] = 0.0;

      testCases[3][0] = 1.0;
      testCases[3][1] = 1.0;
   } // void ABCpopulateTestCases()

/**
 * Populates the truth table from the truth table file
 */
   void populateTruthTable()
   {
      try
      {
         Scanner in = new Scanner(new File(truthTableFile));

         for (int testCase = 0; testCase < NUM_TEST_CASES; testCase++)
            for (int i = 0; i < FNodes; i++)
               truth[testCase][i] = in.nextDouble();

         in.close();
      } // try
      catch (FileNotFoundException fe)
      {
         throw new ValidationError("Truth table file '"+truthTableFile+"' does not exist.");
      } // try catch (FileNotFoundException fe)
   } // void populateTruthTable()

/**
 * Populates the test cases from the inputs file
 */
   void populateTestCases()
   {
      try
      {
         Scanner in = new Scanner(new File(inputsFile));

         for (int testCase = 0; testCase < NUM_TEST_CASES; testCase++)
            for (int k = 0; k < aNodes; k++)
               testCases[testCase][k] = in.nextDouble();

         in.close();
      } //try
      catch (FileNotFoundException fe)
      {
         throw new ValidationError("Inputs file '"+inputsFile+"' does not exist.");
      } // try catch (FileNotFoundException fe)
   } // void populateTestCases()

/**
 * Randomize returns a random double point floating number within the bounds [low, high)
 */
   double randomize(double low, double high)
   {
      return (high - low) * Math.random() + low;
   }

/**
 * Trains or runs the neural network based on the user's selected mode. Activations are calculated with a sigmoid
 * activation function, while training uses gradient descent with changes in weights all applied simultaneously.
 */
   void trainOrRun()
   {
      time = System.currentTimeMillis();

      if (train)
      {
         iterations = 0;
         avgError = Double.MAX_VALUE;                                      //avg error starts above threshold

         while (iterations < maxIterations && avgError > errorThreshold)   //train until max iterations or error threshold reached
         {
            avgError = 0.0;                                                //reset accumulator for the error

            for (int testCase = 0; testCase < NUM_TEST_CASES; testCase++)  //iterate to minimize error in each test case
            {
               runTestCase(testCase);
               calculateGradientDescent(testCase);

               runTestCaseWithoutSaving(testCase);

               for (int i = 0; i < FNodes; i++)
               {
                  double omega_i = truth[testCase][i] - F[i];
                  avgError += omega_i * omega_i / 2.0;
               }
            } // for (int testCase = 0; testCase < NUM_TEST_CASES; testCase++)

            iterations++;
            avgError /= (double) NUM_TEST_CASES;

         } // while (iterations < maxIterations && error < maxAvgError)
      } // if (train)
      for (int testCase = 0; testCase < NUM_TEST_CASES; testCase++)
         runTestCaseWithoutSaving(testCase);

      time = System.currentTimeMillis() - time;
   } // void trainOrRun()

/**
 * Runs the network for one test case, calculating the values of all the hidden nodes and the output node
 * does not save the dot products since they are not needed when only running
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

         h[j] = f.F(theta);
      } // for (int j = 0; j < hNodes; j++) 

      for (int i = 0; i < FNodes; i++)
      {
         theta = 0.0;

         for (int j = 0; j < hNodes; j++)            //iterate over the hidden nodes to calculate their dot product
            theta += h[j] * wji[j][i];

         F[i] = f.F(theta);
         outputs[testCase][i] = F[i];                //store output in outputs array to be later reported
      } // for (int i = 0; i < FNodes; j++)
   } // void runTestCaseWithoutSaving(int testCase)

/**
 * Runs the network for one test case, saving the hidden and output node values, theta js, and psi is
 */
   void runTestCase(int testCase)
   {
      populateInputNodes(testCase);

      for (int j = 0; j < hNodes; j++)
      {
         theta_j[j] = 0.0;

         for (int k = 0; k < aNodes; k++)
            theta_j[j] += a[k] * wkj[k][j];

         h[j] = f.F(theta_j[j]);
      } // for (int j = 0; j < hNodes; j++)

      for (int i = 0; i < FNodes; i++)
      {
         double theta_i = 0.0;

         for (int j = 0; j < hNodes; j++)
            theta_i += h[j] * wji[j][i];

         F[i] = f.F(theta_i);
         outputs[testCase][i] = F[i];

         psi_i[i] = (truth[testCase][i] - F[i]) * f.deriv(theta_i);
      } // for (int i = 0; i < FNodes; i++)
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
 * Calculates the delta weights for the given test case and applies them to the weights
 */
   void calculateGradientDescent(int testCase)
   {
      for (int j = 0; j < hNodes; j++)
      {
         double Omega_j = 0.0;

         for (int i = 0; i < FNodes; i++)
         {
            Omega_j += psi_i[i] * wji[j][i];
            wji[j][i] += lambda * h[j] * psi_i[i];
         }
         
         double Psi_j = Omega_j * f.deriv(theta_j[j]);

         for (int k = 0; k < aNodes; k++)
            wkj[k][j] += lambda * a[k] * Psi_j;
      } // for (int j = 0; j < hNodes; j++)
   } // void calculateGradientDescent(int testCase)

@Deprecated
/**
 * Sigmoid activation function
 */
   double activationFunction(double x)
   {
      return 1.0 / (1.0 + Math.exp(-x));
   }

@Deprecated
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
      System.out.println("Network ran on "+NUM_TEST_CASES+" test cases and took "+time+" milliseconds.\n");

      if (train) reportTrainingInformation();

      System.out.println("Results:\n");      //print the outputs (and truth table)
      if (printTruthTable) reportTruthTable();
      else reportOutputs();

      if (printWeights) reportWeights();
   } // void reportResults() throws FileNotFoundException, IOException

/**
 * Prints out the reasons for training exiting and the iterations and error reached
 */
   void reportTrainingInformation()
   {
      System.out.print("Training exited because the");

      if (iterations >= maxIterations)
         System.out.print(" maximum iterations ("+maxIterations+
               ") was reached");
      if (iterations >= maxIterations && avgError <= errorThreshold)
         System.out.print(" and the");
      if (avgError <= errorThreshold) 
         System.out.print(" error threshold ("+errorThreshold+") was reached");

      System.out.println(".");

      System.out.println(" - Iterations reached: "+iterations);
      System.out.println(" - Average error: "+avgError+"\n");
   } // void reportTrainingInformation

/**
 * Prints the outputs (without the truth table) in a readable format
 */
   void reportOutputs()
   {
      for (int k = 0; k < aNodes; k++)
         System.out.print("a"+k+"  |");

      System.out.println("Outputs");

      for (int k = 0; k < aNodes; k++)
         System.out.print("----|");

      System.out.println("-----------------------------");
      for (int testCase = 0; testCase < NUM_TEST_CASES; testCase++)
      {
         for (int k = 0; k < aNodes; k++)
            System.out.print(testCases[testCase][k]+" |");

         for (int i = 0; i < FNodes; i++)
            System.out.print(format(outputs[testCase][i])+" |");

         System.out.println();
      } // for (int testCase = 0; testCase < NUM_TEST_CASES; testCase++)
   } // void reportOutputs()

/**
 * Prints the truth table and the outputs in a readable format
 */
   void reportTruthTable()
   {
      for (int k = 0; k < aNodes; k++)
      System.out.print("a"+k+"  |");

      System.out.println("Truth         |Outputs");

      for (int k = 0; k < aNodes; k++)
         System.out.print("----|");

      for (int i = 0; i < FNodes; i++)
         System.out.print("----|");

      System.out.println("-----------------------------");

      for (int testCase = 0; testCase < NUM_TEST_CASES; testCase++)
      {
         for (int k = 0; k < aNodes; k++)
            System.out.print(testCases[testCase][k]+" |");

         for (int i = 0; i < FNodes; i++)
            System.out.print(truth[testCase][i]+" |");

         for (int i = 0; i < FNodes; i++)
            System.out.print(format(outputs[testCase][i])+" |");

         System.out.println();
      } // for (int testCase = 0; testCase < NUM_TEST_CASES; testCase++)
   } // void reportTruthTable()

/**
 * Prints the weights in a readable format
 */
   void reportWeights()
   {
      System.out.print("\n\nWeights:");
      for (int j = 0; j < hNodes; j++)
      {
         System.out.print("\n------------------------------------------------------------");
         int k = 0;
         int i = 0;
         while (k < aNodes || i < FNodes)
         {
            if (k < aNodes) 
               System.out.print("\nw1"+k+j+": "+format(wkj[k][j]));
            else
               System.out.print("\n                               ");
            if (i < FNodes) System.out.print("  |w2"+j+i+": "+format(wji[j][i]));

            k++;
            i++;
         } // while (k < aNodes || i < FNodes)
      } // for (int j = 0; j < hNodes; j++)
      System.out.println();   //print a newline for clarity
   } // void reportWeights()

/**
 * Returns a formatted, twenty one character long string for a double so that all the doubles are aligned when displaying them
 */
   String format(double number)
   {
      String numString = ""+Math.abs(number);
      int strLength = numString.length();

      if (strLength < MAX_WEIGHT_STRING_LENGTH)                   //if the weight is less than 20 characters long
      {
         int spaces = MAX_WEIGHT_STRING_LENGTH - strLength;

         for (int count = 0; count < spaces; count++)                      //pad the weight with spaces so they align
            numString += " ";
      }
      if (number > 0) return " "+numString; else return "-"+numString;     //add a space or the negative sign back in

   } // String format(double weight)

/**
 * Main method tests the ABC network on each config file in order, uses "config.txt" if no config files given
 */
   public static void main(String[] args)
   {
      if (args.length == 0)
      {
         String configFile = "config.txt";
         System.out.println("No config file specified, using '"+configFile+"'");
         runNetwork(configFile);
      }
      else
      {
         for (String configFile : args)
            runNetwork(configFile);
      } // if (args.length == 0) else
   } // public static void main(String[] args)

/**
 * Creates an ABC network and executes all the major network arrays in order
 */
   static void runNetwork(String configFile)
   {
      ABCNetwork network = new ABCNetwork();
      try
      {
         network.setConfigurationParameters(configFile);
         network.echoConfigurationParameters();
         network.allocateMemory();
         network.populateArrays();
         network.trainOrRun();
         network.reportResults();
         if (saveWeights) network.saveWeights();
      } // try
      catch (ValidationError e)
      {
         System.out.println("\nError caught while "+origin+".");
         System.out.println(e);
      } // try catch (ValidationError e)
   } // static void runNetwork(String configFile)
} // class AB1Network