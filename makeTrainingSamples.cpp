#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>

using namespace std;

int main()
{
    // Random traning sets for XOR 
    // 2 inputs, 1 output

    ofstream FileOut("trainingData.txt");
    if (!FileOut)
    {
        cout << "Unable to open file!";
        FileOut.close();
        return EXIT_FAILURE;
    }

    FileOut << "topology: 2 5 1" << endl;
    for (int i = 2000; i > 0; --i)
    {
        int n1 = (int)(2.0 * rand() / double(RAND_MAX));
        int n2 = (int)(2.0 * rand() / double(RAND_MAX));
        int t = n1 ^ n2; // XOR outputs 0 or 1
        FileOut << "in: " << n1 << ".0 " << n2 << ".0 " << endl;
        FileOut << "out: " << t << ".0 " << endl;
    } 

    FileOut.close();
    return EXIT_SUCCESS;
}