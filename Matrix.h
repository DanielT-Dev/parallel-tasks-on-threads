#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <iostream>
#include <mutex>
#include <thread>
#include <chrono>
#include <random>
#include <iomanip>
#include <cmath>

using namespace std;

class Matrix {
private:
    int size;
    vector<double> data;

public:
    Matrix(int n);

    int getSize() const;

    double get(int row, int col) const;

    void set(int row, int col, double value);

    void fillRandom(unsigned int seed);

    double checksum() const;

};

class MatrixMultiplier {
private:
    static mutex outputMutex;

public:
    // Computes one element of the resulting matrix.
    static void computeElement(
        const Matrix& A,
        const Matrix& B,
        Matrix& C,
        int row,
        int col,
        int threadId,
        bool debug
    );

    // Returns the execution time in milliseconds.
    template <typename Worker>
    static double measure(
        const Matrix& A,
        const Matrix& B,
        Matrix& C,
        int nrThreads,
        Worker worker
    ) {
        vector<thread> threads;

        auto start = chrono::high_resolution_clock::now();

        for (int i = 0; i < nrThreads; i++) {
            threads.emplace_back(
                worker,
                cref(A),
                cref(B),
                ref(C),
                i,
                nrThreads
            );
        }

        for (thread& t : threads) {
            t.join();
        }

        auto end = chrono::high_resolution_clock::now();

        chrono::duration<double, milli> elapsed = end - start;

        return elapsed.count();
    }
};

#endif
