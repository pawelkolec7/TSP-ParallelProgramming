#include <fstream>
#include <iostream>
#include <string>
#include <random>

int getRandomInt(int a, int b) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(a, b);
    return dis(gen);
}

void generateRandomMatrixWithIntervals(const std::string& filename, int size, int minValue, int maxValue, int intervalWidth) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return;
    }

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            if (i == j) {
                file << "0";
            } else {
                int a = getRandomInt(minValue, maxValue - intervalWidth);
                int b = a + intervalWidth;
                file << a << "-" << b;
            }
            if (j < size - 1) file << " ";
        }
        file << "\n";
    }
    file.close();
}

int main() {
    generateRandomMatrixWithIntervals("matrix.txt", 5000, 10, 40, 8);
    std::cout << "Wygenerowano plik 'matrix.txt'!" << std::endl;
    return 0;
}
