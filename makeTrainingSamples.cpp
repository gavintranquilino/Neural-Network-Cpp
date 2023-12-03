#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>

using namespace std;

int main()
{
    // Random traning sets for XOR 
    // 2 inputs, 1 output

    ofstream FileOut("traningData.txt");
    if (!FileOut)
    {
        cout << "Unable to open file!";
        FileOut.close();
    }

    FileOut << "2 4 1" << endl;
    for (int i = 2000; i > 0; --i)
    {
        int n1 = (int)(2.0 * rand() / double(RAND_MAX));
        int n2 = (int)(2.0 * rand() / double(RAND_MAX));
        int t = n1 ^ n2; // XOR outputs 0 or 1
        FileOut << n1 << ' ' << n2 << ' ' << t << endl;
    } 

    FileOut.close();
    return EXIT_SUCCESS;
}