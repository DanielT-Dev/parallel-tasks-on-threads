# Parallel Matrix Multiplication -- Threading Experiments

## 1. Problem Description

The goal of this assignment is to implement matrix multiplication using
multiple threads and compare different ways of distributing the work
between them.

Given two square matrices, **A** and **B**, of size `N × N`, the program
must compute their product **C**:

\[ C = A `\times `{=tex}B \]

Each element of the resulting matrix is calculated as the dot product of
one row of **A** and one column of **B**:

\[ C\[i\]\[j\] = `\sum`{=tex}\_{k=0}\^{N-1} A\[i\]\[k\]
`\times `{=tex}B\[k\]\[j\] \]

In other words, to calculate one element `C[i][j]`, the program
multiplies corresponding elements from row `i` of matrix **A** and
column `j` of matrix **B**, then adds all the products together.

The important observation is that **each element of the result can be
calculated independently**. Computing `C[0][0]` does not depend on
computing `C[0][1]` or any other element. This makes matrix
multiplication a suitable task for parallel execution: different threads
can calculate different elements of the result matrix at the same time,
without needing to communicate or coordinate while performing the
calculations.

The assignment asks for two levels of functionality:

1.  A function that calculates one element of the resulting matrix.
2.  A thread function that calls the element-calculation function
    repeatedly, calculating the elements assigned to that thread.

The work is divided between a configurable number of threads. Three
distribution strategies are implemented and compared.

### Why compare different strategies?

Although all three strategies perform the same mathematical operations,
they access memory in different orders and assign different elements to
each thread. This can affect performance because of:

-   **Cache locality:** processors load memory in cache lines. Accessing
    nearby memory locations consecutively can make better use of the
    cache.
-   **Thread creation overhead:** starting threads has a cost. For small
    matrices, this cost can be significant compared with the actual
    computation.
-   **Scheduling and workload distribution:** threads should receive
    approximately equal amounts of work. Different distribution methods
    can affect how evenly the work is divided.
-   **Memory access patterns:** the order in which matrix elements are
    read and written can influence cache performance.

The objective is therefore not only to calculate the product correctly,
but also to observe how the distribution of independent tasks influences
execution time.

## 2. Implementation

### 2.1. Calculating one element

The first function is responsible for calculating exactly one element of
the result matrix.

It receives the row and column of the element to calculate, together
with the input matrices. It iterates through the corresponding row of
the first matrix and column of the second matrix, accumulating the
products.

For debugging, it also prints the coordinates of the element and the ID
of the thread that is calculating it.

``` cpp
void computeElement(
    const Matrix& a,
    const Matrix& b,
    Matrix& result,
    int row,
    int col,
    int threadId
) {
    double sum = 0.0;

    for (int k = 0; k < a.size; ++k) {
        sum += a(row, k) * b(k, col);
    }

    result(row, col) = sum;

    cout << "Thread " << threadId
         << " computes element (" << row
         << ", " << col << ")\n";
}
```

Each invocation calculates one complete result element. Since each
thread writes to a different position in the result matrix, the
calculations do not require a mutex or other synchronization mechanism,
provided that the work distribution does not assign the same element to
multiple threads.

### 2.2. Distributing consecutive elements in row-major order

In the first strategy, the result matrix is treated as a one-dimensional
sequence of `N × N` elements, traversed row by row.

For example, a `3 × 3` matrix is traversed in this order:

``` text
(0,0) (0,1) (0,2)
(1,0) (1,1) (1,2)
(2,0) (2,1) (2,2)
```

The total number of elements is divided as evenly as possible between
the threads. Each thread receives a consecutive interval of this
sequence.

The following example illustrates the calculation of a thread's assigned
interval. The `start` index is inclusive, while `end` is exclusive.

``` cpp
int totalElements = matrixSize * matrixSize;
int baseCount = totalElements / numberOfThreads;
int remainder = totalElements % numberOfThreads;

int count = baseCount + (threadId < remainder ? 1 : 0);
int start = threadId * baseCount + min(threadId, remainder);
int end = start + count;

for (int index = start; index < end; ++index) {
    int row = index / matrixSize;
    int col = index % matrixSize;

    computeElement(a, b, result, row, col, threadId);
}
```

The division and modulo operations convert a one-dimensional index back
into a row and column.

For a `9 × 9` matrix and four threads, there are 81 elements. The work
is distributed as 21, 20, 20, and 20 elements respectively. Each thread
processes its assigned elements consecutively, even when its interval
crosses a row boundary.

### 2.3. Distributing consecutive elements in column-major order

The second strategy is similar to the first one, but traverses the
result matrix column by column instead of row by row.

For a `3 × 3` matrix, the traversal order is:

``` text
(0,0) (1,0) (2,0)
(0,1) (1,1) (2,1)
(0,2) (1,2) (2,2)
```

The same division of the total number of elements is used. However, the
one-dimensional index is converted into coordinates differently:

``` cpp
int totalElements = matrixSize * matrixSize;
int baseCount = totalElements / numberOfThreads;
int remainder = totalElements % numberOfThreads;

int count = baseCount + (threadId < remainder ? 1 : 0);
int start = threadId * baseCount + min(threadId, remainder);
int end = start + count;

for (int index = start; index < end; ++index) {
    int row = index % matrixSize;
    int col = index / matrixSize;

    computeElement(a, b, result, row, col, threadId);
}
```

This strategy assigns consecutive elements in column-major order. It is
useful for comparing the effect of changing the traversal direction
while keeping the workload approximately equal.

### 2.4. Cyclic distribution

In the third strategy, the threads do not receive consecutive intervals.
Instead, each thread calculates every `k`-th element, where `k` is the
number of threads.

The matrix is traversed in row-major order. Thread 0 calculates indices
`0, k, 2k, ...`; thread 1 calculates indices `1, k+1, 2k+1, ...`, and so
on.

``` cpp
int totalElements = matrixSize * matrixSize;

for (int index = threadId;
     index < totalElements;
     index += numberOfThreads) {

    int row = index / matrixSize;
    int col = index % matrixSize;

    computeElement(a, b, result, row, col, threadId);
}
```

For example, with four threads, thread 0 calculates indices 0, 4, 8, 12,
and so on, while thread 1 calculates indices 1, 5, 9, 13, and so on.

This approach distributes the elements evenly across the threads. Unlike
the first two strategies, however, each thread jumps between different
parts of the matrix instead of processing one consecutive interval.

### 2.5. Creating and joining the threads

For each strategy, the main program creates the requested number of
threads and passes each thread its ID, the total number of threads, and
references to the input and result matrices.

The following is a simplified example of the thread creation and
synchronization process:

``` cpp
vector<thread> threads;

for (int threadId = 0; threadId < numberOfThreads; ++threadId) {
    threads.emplace_back(
        workerFunction,
        cref(a),
        cref(b),
        ref(result),
        threadId,
        numberOfThreads
    );
}

for (thread& t : threads) {
    t.join();
}
```

The `join()` calls make the main thread wait until all worker threads
have finished. Only after all threads have completed is the resulting
matrix considered fully computed.

The actual worker function selects the appropriate distribution strategy
and calls `computeElement()` for each assigned position.

### 2.6. Measuring execution time

Execution time is measured around the parallel computation. The timer
should start immediately before the threads are created and stop after
all threads have been joined. This includes both thread creation and
thread execution, allowing the experiment to capture the overhead of
using multiple threads.

Debug printing should be disabled during performance measurements.
Printing a message for every computed element introduces substantial I/O
overhead and can distort the results, making it difficult to compare the
actual computation times.

## 3. Experimental Results

The following configurations were tested for each strategy. The matrix
size and number of threads were increased between configurations to
observe how the execution time changes as the workload grows.

### STRATEGY 1: ROW-MAJOR CONSECUTIVE

Each thread computes a consecutive range of elements, traversing the
result matrix row by row.

**Configuration 1** - Matrix size: 100 × 100 - Threads: 2 - Execution
time: 7.732 ms

**Configuration 2** - Matrix size: 500 × 500 - Threads: 4 - Execution
time: 426.863 ms

**Configuration 3** - Matrix size: 1000 × 1000 - Threads: 8 - Execution
time: 2113.551 ms

### STRATEGY 2: COLUMN-MAJOR CONSECUTIVE

Each thread computes a consecutive range of elements, traversing the
result matrix column by column.

**Configuration 1** - Matrix size: 100 × 100 - Threads: 2 - Execution
time: 6.444 ms

**Configuration 2** - Matrix size: 500 × 500 - Threads: 4 - Execution
time: 318.884 ms

**Configuration 3** - Matrix size: 1000 × 1000 - Threads: 8 - Execution
time: 1582.752 ms

### STRATEGY 3: CYCLIC DISTRIBUTION

Each thread computes every *k*-th element in row-major order, where *k*
is the number of threads.

**Configuration 1** - Matrix size: 100 × 100 - Threads: 2 - Execution
time: 6.445 ms

**Configuration 2** - Matrix size: 500 × 500 - Threads: 4 - Execution
time: 268.066 ms

**Configuration 3** - Matrix size: 1000 × 1000 - Threads: 8 - Execution
time: 1574.857 ms

## 4. Brief Comparison

For all three tested configurations, the column-major and cyclic
strategies were faster than the row-major consecutive strategy. The
cyclic strategy achieved the lowest execution time for the two larger
matrices, while column-major was slightly faster for the smallest
matrix.

  Matrix size     Threads     Row-major   Column-major        Cyclic
  ------------- --------- ------------- -------------- -------------
  100 × 100             2      7.732 ms       6.444 ms      6.445 ms
  500 × 500             4    426.863 ms     318.884 ms    268.066 ms
  1000 × 1000           8   2113.551 ms    1582.752 ms   1574.857 ms

The results suggest that the order in which work is distributed can
influence performance, even though all strategies perform the same
number of arithmetic operations.

However, these measurements alone do not prove that one strategy is
universally faster. Results can depend on the matrix representation,
cache behavior, processor architecture, compiler optimizations, and
system load. Repeating each experiment and calculating an average would
provide a more reliable comparison.

Overall, the experiments demonstrate that dividing a task between
threads is not only about assigning an equal number of operations to
each thread. Memory access patterns and the overhead introduced by
parallel execution can also have a significant impact on performance.
