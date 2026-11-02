#include "Matrix.h"

const bool DEBUG = false;

// Thread function.
void rowMajorWorker(
    const Matrix& A,
    const Matrix& B,
    Matrix& C,
    int threadId,
    int nrThreads
) {
    int n = A.getSize();
    int totalElements = n * n;

    int start = totalElements * threadId / nrThreads;
    int end = totalElements * (threadId + 1) / nrThreads;

    for (int index = start; index < end; index++) {

        int row = index / n;
        int col = index % n;

        MatrixMultiplier::computeElement(
            A, B, C, row, col, threadId, DEBUG
        );
    }
}

int main() {
    vector<pair<int, int>> configurations = {
        { 100, 2 },
        { 500, 4 },
        { 1000, 8 }
    };

    cout << fixed << setprecision(3);

    cout << "STRATEGY 1: ROW-MAJOR CONSECUTIVE\n";

    for (int i = 0; i < (int)configurations.size(); i++) {

        int n = configurations[i].first;
        int nrThreads = configurations[i].second;

        Matrix A(n);
        Matrix B(n);
        Matrix C(n);

        A.fillRandom(123);
        B.fillRandom(456);

        double elapsed = MatrixMultiplier::measure(
            A, B, C, nrThreads, rowMajorWorker
        );

        cout << "\nConfiguration " << i + 1 << endl;
        cout << "Matrix size: " << n << " x " << n << endl;
        cout << "Threads: " << nrThreads << endl;
        cout << "Execution time: " << elapsed << " ms" << endl;
        cout << "Checksum: " << C.checksum() << endl;
    }

    return 0;
}
