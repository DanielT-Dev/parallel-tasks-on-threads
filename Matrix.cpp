#include "Matrix.h"

mutex MatrixMultiplier::outputMutex;

Matrix::Matrix(int n) : size(n), data(n* n, 0.0) {}

int Matrix::getSize() const {
    return size;
}

double Matrix::get(int row, int col) const {
    return data[row * size + col];
}

void Matrix::set(int row, int col, double value) {
    data[row * size + col] = value;
}

void Matrix::fillRandom(unsigned int seed) {
    mt19937 generator(seed);
    uniform_real_distribution<double> distribution(1.0, 10.0);

    for (double& value : data) {
        value = distribution(generator);
    }
}

double Matrix::checksum() const {
    double sum = 0.0;

    for (double value : data) {
        sum += value;
    }

    return sum;
}

void MatrixMultiplier::computeElement(
    const Matrix& A,
    const Matrix& B,
    Matrix& C,
    int row,
    int col,
    int threadId,
    bool debug
) {
    int n = A.getSize();
    double sum = 0.0;

    for (int k = 0; k < n; k++) {
        sum += A.get(row, k) * B.get(k, col);
    }

    C.set(row, col, sum);

    if (debug) {
        lock_guard<mutex> lock(outputMutex);

        cout << "Thread " << threadId
            << " computed element ("
            << row << ", " << col << ")"
            << endl;
    }
}
