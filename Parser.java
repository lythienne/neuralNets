import java.util.Scanner;

/**
 * Author: Harrison Chen
 * Date of Creation: 3/12/24
 * 
 * Parser takes in one line of the config file and extracts its value
 * 
 * Table of Contents:
 *    static String cleanString(String string)
 *    static boolean setString(String line)
 *    static int extractInt()
 *    static int extractDouble()
 *    static int extractBoolean()
 *    static int extractString()
 */
class Parser
{
   static Scanner in;         //holds the line from the config file and extracts the value from it
   static String value;       //stores the value from the line

/**
 * Returns the text of the string before the equals sign
 */
   static String cleanString(String string)
   {
      return string.substring(0, string.indexOf("=")).strip();
   }

/**
 * Sets the line in the config file for this parser to extract values from
 * Returns true if a value is found in the line, false otherwise
 */
   static boolean setString(String line)
   {
      boolean valueFound = true;

      in = new Scanner(line);

      try
      {
         value = line.substring(line.indexOf("=") + 1, line.indexOf(";"));    //+1 so the value is only whats within the parentheses
         value = value.strip();
      }
      catch (IndexOutOfBoundsException e)
      {
         valueFound = false;
      }  // try catch (IndexOutOfBoundsException e)

      return valueFound;
   } // boolean setString(String line)

/**
 * Extracts the integer from the line in the config file
 */
   static int extractInt()
   {
      return Integer.parseInt(value);
   }

/**
 * Extracts the double from the line in the config file
 */
   static double extractDouble()
   {
      return Double.parseDouble(value);
   }

/**
 * Extracts the boolean from the line in the config file
 */
   static boolean extractBoolean()
   {
      return Boolean.parseBoolean(value);
   }

/**
 * Extracts the string from the line in the config file
 */
   static String extractString()
   {
      return value;
   }
} // class Parser
