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
 * The NLayerNetwork class is an n-layer multilayer perceptron with backpropagation.
 * 
 * Table of Contents:
 *    setConfigurationParameters(String)
 *       parseLine(String)
 *    echoConfigurationParameters()
 *       echoValidateLoadWeights()
 *    allocateMemory()
 *    populateArrays()
 *       populateWeightsManually()
 *       saveWeights()
 *       loadWeights()
 *       AB1populateTruthTable()
 *       AB1populateTestCases()
 *       populateTruthTable()
 *       populateTestCases()
 *       randomize(double, double)
 *    trainOrRun()
 *       runTestCaseWithoutSaving(int)
 *       runTestCase(int)
 *       populateInputNodes(int)
 *       calculateGradientDescent(int)
 *       activationFunction(double)
 *       derivative(double)
 *    reportResults()
 *       reportTrainingInformation()
 *       reportOutputs()
 *       reportTruthTable()
 *       reportWeights()
 *       format(double)
 *    main(String[])
 *       runNetwork(String)
 */
class NLayerNetwork
{
   final static int A = 0;                            //layer of the input layer

   final static int RANDOMIZE = 0;                    //randomly populates the weights
   final static int TWO_TWO_ONE = 1;                  //populates the weights with preset values for a 2-2-1 network
   final static int LOAD_FROM_FILE = 2;               //populates the weights from a file

   final static int[] PRESET_MODEL = {2, 2, 1};       //array constant that saves a 2-2-1 model to be checked against
   final static int PRESET_LAYERS = 2;                //number of layers in the preset model
   final static int PRESET_A_INDEX = 0;               //index of the number of input nodes in the preset model
   final static int PRESET_H_INDEX = 1;               //index of the number of hidden nodes in the preset model
   final static int PRESET_F_INDEX = 2;               //index of the number of final node in the preset model

   final static int a0INDEX = 0;                      //index of the first input value in the test cases array
   final static int a1INDEX = 1;                      //index of the second input value in the test cases array

   final static double MS_THRESHOLD = 1.0;            //if the number of seconds is less than one then print in milliseconds
   final static double MS_PER_SEC = 1000.0;           //number of milliseconds per second
   final static double SEC_PER_MIN = 60.0;            //number of seconds per minute
   final static double MIN_PER_HR = 60.0;             //number of minutes per hour
   final static double HR_PER_DAY = 24.0;             //number of hours per day
   final static double DAY_PER_WK = 7.0;              //number of days per week

   final static int MAX_WEIGHT_STRING_LENGTH = 24;    //the maximum length the weight can be when printed so that they all line up

   int layers;                   //number of network (connectivity) layers
   int nodeLayers;               //number of node layers

   int F;                        //index of the output layer

   int[] numNodes;               //number of nodes in each layer
   double[][] a;                 //2d array for all the activations in the network

   boolean train;                //whether to train (true) or run (false)

   double errorThreshold;        //maximum average error for training to end
   int maxIterations;            //maximum number of training cycles before training times out
   double time;                  //stores the time taken to complete running/training

   int weightPopulation;         //weight population method (randomized, 2-2-1 preset, or loaded from file)
   double randomLow;             //randomized weight low bound (inclusive)
   double randomHigh;            //randomized weight high bound (exclusive)

   double lambda;                //learning factor
   ActivationFunction f;         //the activation function used to calculate the values of the activations

   double[][][] w;               //3d weights array for the whole network

   double[][] theta;             //2d dot products array
   double[][] psi;               //2d array for the psi intermediate values

   int numTestCases;             //the number of test cases

   int iterations;               //number of iterations in the current training session
   double avgError;              //average error from the previous iteration

   int keepAlive;                //number of iterations after which to print a keep-alive message
   boolean printKeepAlives;      //whether or not to print keep-alive messages

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
   String inputsFile;            //the file where the inputs are stored

/**
 * Values for the configuration parameters will be read from the config file here.
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

      nodeLayers = layers + 1;               //node layers is the number of connectivity layers plus one
      F = layers;                            //index of the output layer is equal to the number of connectivity layers
   } // void setConfigurationParameters(String configFile)

/**
 * Parses a line in the config file to assign the variable in the line to the value in the line
 */
   void parseLine(String line)
   {
      switch (line)
      {
         case "network": numNodes = Parser.extractConnectivityPattern(); break;
         case "layers": layers = Parser.extractInt(); break;

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

         case "numTestCases": numTestCases = Parser.extractInt(); break;

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
         case "keepAlive": keepAlive = Parser.extractInt(); break;
         case "printKeepAlives" : printKeepAlives = Parser.extractBoolean(); break;

         case "randomLow": randomLow = Parser.extractDouble(); break;
         case "randomHigh": randomHigh = Parser.extractDouble(); break;

         case "saveWeightFile": saveWeightFile = Parser.extractString(); break;
         case "loadWeightFile": loadWeightFile = Parser.extractString(); break;
         case "truthTableFile": truthTableFile = Parser.extractString(); break;
         case "inputsFile": inputsFile = Parser.extractString(); break;

      } // switch (line)
   } // void parseLine(String line)

/**
 * Validates and echoes back all the configuration parameters to the user in an easily digestible format.
 * 
 * Throws ValidationErrors if the user selected nonsensical parameters or parameters that do not match the model.
 */
   void echoConfigurationParameters()
   {
      origin = "echoing and validating the configuration parameters";

      System.out.println("-----------------------------------------------------------");

      if (nodeLayers != numNodes.length)     //use .length to check if number of layers matches the connectivity pattern in config
      {
         throw new ValidationError("'"+layers+"' connectivity layers/'"+nodeLayers+
               "' node layers declared in config file does not match connectivity pattern.");
      }

      System.out.print("Configuration Parameters for a ");

      System.out.print(numNodes[0]);         //print first element before looping to make it look nice
      for (int n = 1; n < nodeLayers; n++)
      {
         System.out.print("-"+numNodes[n]);
      }

      System.out.print(" network:\n\n");
      
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
            if (layers != PRESET_LAYERS || PRESET_MODEL[PRESET_A_INDEX] != numNodes[PRESET_A_INDEX]
                     || PRESET_MODEL[PRESET_H_INDEX] != numNodes[PRESET_H_INDEX] 
                     || PRESET_MODEL[PRESET_F_INDEX] != numNodes[PRESET_F_INDEX])
               throw new ValidationError("Preset model selected and network connectivity pattern does not match preset 2-2-1 model.");
            break;

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

      System.out.println();   //print a line for clarity

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
         if (printKeepAlives)
            System.out.println("\nPrinting keep-alives every "+keepAlive+" iterations.");
      } // if (train)
      else
      {
         System.out.println("Network is running");
      } // if (train) else

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

      for (int n = 0; n < nodeLayers; n++)
      {
         int layerNnodes = (int) in.readDouble();
         if (layerNnodes != numNodes[n])
            throw new ValidationError("Saved number of nodes in layer "+n+": "+layerNnodes+
               "' in weights file '"+loadWeightFile+"' does not match user set: "+numNodes[n]+" in config file");
      } // for (int n = 0; n < nodeLayers; n++)

      in.close();

      System.out.println("Loading weights from '"+loadWeightFile+"'");
   } // void echoValidateLoadWeights() throws FileNotFoundException, IOException

/**
 * Allocates memory for all major network arrays such as the activations, weights, test cases, and truth table.
 */
   void allocateMemory()
   {
      origin = "allocating memory for the arrays";

      a = new double[nodeLayers][];                //number of activation layers is equal to the number of network layers + 1

      for (int n = 0; n < nodeLayers; n++)
         a[n] = new double[numNodes[n]];

      w = new double[layers][][];

      for (int n = 0; n < layers; n++)
         w[n] = new double[numNodes[n]][numNodes[n+1]];

      truth = new double[numTestCases][numNodes[F]];
      testCases = new double[numTestCases][numNodes[A]];

      outputs = new double[numTestCases][numNodes[F]];

      if (train)
      {
         theta = new double[nodeLayers][];          //thetas will be 1-indexed so the indexes match up
         for (int n = 1; n < nodeLayers; n++)
            theta[n] = new double[numNodes[n]];

         psi = new double[nodeLayers][];            //psis will be 1-indexed so the indexes match up
         for (int n = 1; n < nodeLayers; n++)
            psi[n] = new double[numNodes[n]];
      } // if (train)
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
            for (int n = 0; n < layers; n++)
               for (int m = 0; m < numNodes[n]; m++)
                  for (int k = 0; k < numNodes[n + 1]; k++)
                     w[n][m][k] = randomize(randomLow, randomHigh);
            break; // case RANDOMIZE:
            
         case LOAD_FROM_FILE: loadWeights(); break;

         case TWO_TWO_ONE: populateWeightsManually(); break;

         default:
            throw new ValidationError("Please select a valid weight population method, "+
                  "also check your echo/validate since this should have been caught there");
      } // switch (weightPopulation)

      populateTestCases();
      populateTruthTable();

   } // void populateArrays()

/**
 * Manually populates weights for a 2-2-1 network into the weights arrays
 * 
 * Will always fail for ABCD networks, will reimplement in n-layer
 */
   void populateWeightsManually()
   {
      w[PRESET_A_INDEX][0][0] = 0.2;
      w[PRESET_A_INDEX][0][1] = 0.243;
      w[PRESET_A_INDEX][1][0] = 0.1;
      w[PRESET_A_INDEX][1][1] = 0.353;

      w[PRESET_H_INDEX][0][0] = 0.75;
      w[PRESET_H_INDEX][1][0] = 0.757;

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

         for (int n = 0; n < nodeLayers; n++)
            out.writeDouble(numNodes[n]);

         for (int n = 0; n < layers; n++)
            for (int m = 0; m < numNodes[n]; m++)
               for (int k = 0; k < numNodes[n + 1]; k++)
                  out.writeDouble(w[n][m][k]);
         
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

         for (int n = 0; n < nodeLayers; n++)      //skips the doubles which are the network connectivity model
            in.readDouble();

         for (int n = 0; n < layers; n++)
            for (int m = 0; m < numNodes[n]; m++)
               for (int k = 0; k < numNodes[n + 1]; k++)
                  w[n][m][k] = in.readDouble();
         
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

   } // void AB1populateTruthTable()

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

         for (int testCase = 0; testCase < numTestCases; testCase++)
            for (int i = 0; i < numNodes[F]; i++)
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

         for (int testCase = 0; testCase < numTestCases; testCase++)
            for (int k = 0; k < numNodes[A]; k++)
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
 * Trains or runs the neural network based on the user's selected mode. Activations are calculated with a user declared
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

            for (int testCase = 0; testCase < numTestCases; testCase++)    //iterate to minimize error in each test case
            {
               populateInputNodes(testCase);
               runTestCase(testCase);
               calculateGradientDescent(testCase);

               runTestCaseWithoutSaving(testCase);

               for (int i = 0; i < numNodes[F]; i++)
               {
                  double omega_i = truth[testCase][i] - a[F][i];
                  avgError += omega_i * omega_i / 2.0;
               }
            } // for (int testCase = 0; testCase < numTestCases; testCase++)

            iterations++;
            avgError /= (double) numTestCases;
            
            if (printKeepAlives && iterations % keepAlive == 0)
               System.out.println("Iterations: "+iterations+", Error: "+avgError);

         } // while (iterations < maxIterations && error < maxAvgError)
      } // if (train)
      
      for (int testCase = 0; testCase < numTestCases; testCase++)
      {
         populateInputNodes(testCase);
         runTestCaseWithoutSaving(testCase);
      }

      time = System.currentTimeMillis() - time;
   } // void trainOrRun()

/**
 * Runs the network for one test case, calculating the values of all the hidden nodes and the output node
 * does not save the thetas or psis since they are not needed when only running
 */
   void runTestCaseWithoutSaving(int testCase)
   {
      double theta;

      for (int n = 1; n < layers; n++)    //layers is the number of node layers - 1 since we don't want to loop over the output layer twice
      {
         for (int k = 0; k < numNodes[n]; k++)
         {
            theta = 0.0;

            for (int m = 0; m < numNodes[n - 1]; m++)
               theta += a[n - 1][m] * w[n - 1][m][k];

            a[n][k] = f.F(theta);
         } // for (int k = 0; k < numNodes[n]; k++)
      } // for (int n = 1; n < layers; n++)
      
      int n = F;
      for (int k = 0; k < numNodes[n]; k++)
      {
         theta = 0.0;

         for (int m = 0; m < numNodes[n - 1]; m++)   //iterate over the hidden nodes to calculate their dot product
            theta += a[n - 1][m] * w[n - 1][m][k];

         a[n][k] = f.F(theta);
         outputs[testCase][k] = a[n][k];             //store output in outputs array to be later reported
      } // for (int k = 0; k < numNodes[n]; k++)
   } // void runTestCaseWithoutSaving(int testCase)

/**
 * Runs the network for one test case, saving the hidden and output node values, thetas, and psis
 */
   void runTestCase(int testCase)
   {
      for (int n = 1; n < layers; n++)
      {
         for (int k = 0; k < numNodes[n]; k++)
         {
            theta[n][k] = 0.0;

            for (int m = 0; m < numNodes[n - 1]; m++)
               theta[n][k] += a[n - 1][m] * w[n - 1][m][k];

            a[n][k] = f.F(theta[n][k]);
         } // for (int k = 0; k < numNodes[n]; k++)
      } // for (int n = 1; n < layers; n++)

      int n = F;
      for (int k = 0; k < numNodes[n]; k++)
      {
         theta[n][k] = 0.0;

         for (int m = 0; m < numNodes[n - 1]; m++)
            theta[n][k] += a[n - 1][m] * w[n - 1][m][k];

         a[n][k] = f.F(theta[n][k]);

         outputs[testCase][k] = a[n][k];
         psi[n][k] = (truth[testCase][k] - a[n][k]) * f.deriv(theta[n][k]);
      } // for (int k = 0; k < numNodes[n]; k++)
   } // void runTestCase(int testCase)

/**
 * Populates the input nodes with the correct values for the given test case
 */
   void populateInputNodes(int testCase)
   {
      for (int m = 0; m < numNodes[A]; m++)
         a[A][m] = testCases[testCase][m];
   }

/**
 * Calculates the delta weights for the given test case and applies them to the weights
 */
   void calculateGradientDescent(int testCase)
   {
      for (int n = F - 1; n > 1; n--)           //start in second to last layer
      {
         for (int j = 0; j < numNodes[n]; j++)
         {
            double Omega_j = 0.0;               //initialize accumulator

            for (int i = 0; i < numNodes[n + 1]; i++)
            {
               Omega_j += psi[n + 1][i] * w[n][j][i];
               w[n][j][i] += lambda * a[n][j] * psi[n + 1][i];
            }

            psi[n][j] = Omega_j * f.deriv(theta[n][j]);
         } // for (int j = 0; j < numNodes[n]; j++)
      } // for (int n = F - 1; n > 1; n--)

      int n = A + 1;
      for (int j = 0; j < numNodes[n]; j++)
      {
         double Omega_j = 0.0;               //initialize accumulator

         for (int i = 0; i < numNodes[n + 1]; i++)
         {
            Omega_j += psi[n + 1][i] * w[n][j][i];
            w[n][j][i] += lambda * a[n][j] * psi[n + 1][i];
         }

         psi[n][j] = Omega_j * f.deriv(theta[n][j]);
         
         for (int m = 0; m < numNodes[n - 1]; m++)
            w[n - 1][m][j] += lambda * a[n - 1][m] * psi[n][j];
      } // for (int j = 0; j < numNodes[n]; j++)
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
      if (printKeepAlives)
         System.out.println("-----------------------------------------------------------");
   
      System.out.print("Network ran on "+numTestCases+" test cases and took ");
      printTime(time);

      if (train) reportTrainingInformation();

      System.out.println("Results:\n");      //print the outputs (and truth table)
      if (printTruthTable) reportTruthTable();
      else reportOutputs();

      if (printWeights) reportWeights();
   } // void reportResults()

/**
 * Accept a value representing milliseconds elapsed and print out a decimal value in easier to digest units
 */
void printTime(double milliseconds)
{
   double seconds, minutes, hours, days, weeks;

   seconds = milliseconds / MS_PER_SEC;

   if (seconds < MS_THRESHOLD)
      System.out.printf("%g milliseconds", seconds * MS_PER_SEC);
   else if (seconds < SEC_PER_MIN)
      System.out.printf("%g seconds", seconds);
   else
   {
      minutes = seconds / SEC_PER_MIN;

      if (minutes < SEC_PER_MIN)
         System.out.printf("%g minutes", minutes);
      else
      {
         hours = minutes / MIN_PER_HR;

         if (hours < HR_PER_DAY)
            System.out.printf("%g hours", hours);
         else
         {
            days = hours / HR_PER_DAY;

            if (days < DAY_PER_WK)
               System.out.printf("%g days", days);
            else
            {
               weeks = days / DAY_PER_WK;

               System.out.printf("%g weeks", weeks);
            }
         } // if (hours < HR_PER_DAY) else
      } // if (minutes < MIN_PER_HR) else
   } // else if (seconds < SEC_PER_MIN) else

   System.out.printf("\n\n");
} // void printTime(double milliseconds)

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
   } // void reportTrainingInformation()

/**
 * Prints the outputs (without the truth table) in a readable format
 */
   void reportOutputs()
   {
      for (int k = 0; k < numNodes[A]; k++)
         System.out.print("a"+k+"  |");

      System.out.println("Outputs");

      for (int k = 0; k < numNodes[A]; k++)
         System.out.print("----|");

      System.out.println("-----------------------------");
      for (int testCase = 0; testCase < numTestCases; testCase++)
      {
         for (int k = 0; k < numNodes[A]; k++)
            System.out.print(testCases[testCase][k]+" |");

         for (int i = 0; i < numNodes[F]; i++)
            System.out.print(format(outputs[testCase][i])+" |");

         System.out.println();
      } // for (int testCase = 0; testCase < numTestCases; testCase++)
   } // void reportOutputs()

/**
 * Prints the truth table and the outputs in a readable format
 */
   void reportTruthTable()
   {
      for (int k = 0; k < numNodes[A]; k++)
      System.out.print("a"+k+"  |");

      System.out.println("|Truth         ||Outputs");

      for (int k = 0; k < numNodes[A]; k++)
         System.out.print("----|");

      System.out.print("|");

      for (int i = 0; i < numNodes[F]; i++)
         System.out.print("----|");

      System.out.println("|-----------------------------");

      for (int testCase = 0; testCase < numTestCases; testCase++)
      {
         for (int k = 0; k < numNodes[A]; k++)
            System.out.print(testCases[testCase][k]+" |");

         System.out.print("|");

         for (int i = 0; i < numNodes[F]; i++)
            System.out.print(truth[testCase][i]+" |");

         System.out.print("|");

         for (int i = 0; i < numNodes[F]; i++)
            System.out.print(format(outputs[testCase][i])+" |");

         System.out.println();
      } // for (int testCase = 0; testCase < numTestCases; testCase++)
   } // void reportTruthTable()

/**
 * Prints the weights in a readable format
 */
   void reportWeights()
   {
      System.out.print("\n\nWeights:");

      for (int n = 0; n < layers; n++)
      {
         for (int m = 0; m < numNodes[n]; m++)
            for (int k = 0; k < numNodes[n + 1]; k++)
               System.out.println("w"+n+m+k+": "+format(w[n][m][k]));
         System.out.print("\n------------------------------------------------------------");
      }
   } // void reportWeights()

/**
 * Returns a formatted, twenty four character long string for a double so that all the doubles are aligned when displaying them
 */
   String format(double number)
   {
      String numString = ""+Math.abs(number);
      int strLength = numString.length();

      if (strLength < MAX_WEIGHT_STRING_LENGTH)                         //if the weight is less than 20 characters long
      {
         int spaces = MAX_WEIGHT_STRING_LENGTH - strLength;

         for (int count = 0; count < spaces; count++)                   //pad the weight with spaces so they align
            numString += " ";
      }
      if (number > 0) return " "+numString; else return "-"+numString;  //add a space or the negative sign back in

   } // String format(double number)

/**
 * Main method tests the network on each config file in order, uses "config.txt" if no config files given
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
 * Creates a network and executes all the major network arrays in order
 */
   static void runNetwork(String configFile)
   {
      NLayerNetwork network = new NLayerNetwork();
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
} // class NLayerNetwork