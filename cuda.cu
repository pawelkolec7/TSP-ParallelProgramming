#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <chrono>
#include <cuda.h>
#include <curand_kernel.h>

#define MAX_GENERATIONS 10
#define MUTATION_RATE 0.05f
#define ELITISM 1
#define THREADS_PER_BLOCK 256

struct CostRange {
    int min_cost;
    int max_cost;
};

struct Chromosome {
    int* order; 
    int fitness;
};

__host__ __device__
int mod(int x, int N) {
    return x % N;
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

std::vector<std::vector<CostRange>> readMatrixFromFile(const std::string& filename, int N) {
    std::vector<std::vector<CostRange>> matrix(N, std::vector<CostRange>(N));
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Can't open file: " << filename << std::endl;
        return {};
    }
    std::string s;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            file >> s;
            if (s == "0") {
                matrix[i][j] = {0, 0};
            } else {
                size_t dash = s.find('-');
                int min_c = std::stoi(s.substr(0, dash));
                int max_c = std::stoi(s.substr(dash + 1));
                matrix[i][j] = {min_c, max_c};
            }
        }
    }
    return matrix;
}

void flattenMatrix(const std::vector<std::vector<CostRange>>& matrix, CostRange* d_matrix, int N) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            d_matrix[i*N + j] = matrix[i][j];
}


__global__ void setup_kernel(curandState *state, unsigned long seed, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n)
        curand_init(seed, id, 0, &state[id]);
}

__device__ int random_cost(const CostRange& cr, curandState* state, int tid) {
    if (cr.min_cost == cr.max_cost) return cr.min_cost;
    int diff = cr.max_cost - cr.min_cost + 1;
    int r = curand(&state[tid]) % diff;
    return cr.min_cost + r;
}

__device__ int compute_fitness_gpu(int* order, const CostRange* matrix, int N, curandState* state, int tid) {
    int sum = 0;
    for (int i = 0; i < N; ++i) {
        int from = order[i];
        int to = order[(i + 1) % N];
        CostRange cr = matrix[from * N + to];
        sum += random_cost(cr, state, tid);
    }
    return sum;
}

__global__ void evaluate_population_kernel(
    int* d_orders, int* d_fitness,
    const CostRange* d_matrix, int N, int pop_size, curandState* state)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < pop_size) {
        int* chromo = d_orders + idx*N;
        int fit = compute_fitness_gpu(chromo, d_matrix, N, state, idx);
        d_fitness[idx] = fit;
    }
}

__global__ void mutate_kernel(int* d_orders, int N, int pop_size, float mutation_rate, curandState* state) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < pop_size) {
        int* chromo = d_orders + idx*N;
        for (int i = 0; i < N; ++i) {
            float p = curand_uniform(&state[idx]);
            if (p < mutation_rate) {
                int j = curand(&state[idx]) % N;
                int tmp = chromo[i];
                chromo[i] = chromo[j];
                chromo[j] = tmp;
            }
        }
    }
}


void tournament_selection(const int* orders, const int* fitness, int pop_size, int N, std::mt19937& rng, int* selected) {
    std::uniform_int_distribution<int> dist(0, pop_size-1);
    int a = dist(rng);
    int b = dist(rng);
    int winner = (fitness[a] < fitness[b]) ? a : b;
    for (int i = 0; i < N; ++i)
        selected[i] = orders[winner*N + i];
}

void ox_crossover(const int* parent1, const int* parent2, int* child, int N, std::mt19937& rng) {
    std::uniform_int_distribution<int> dist(0, N-1);
    int start = dist(rng);
    int end = dist(rng);
    if (start > end) std::swap(start, end);

    std::fill(child, child+N, -1);

    for (int i = start; i <= end; ++i)
        child[i] = parent1[i];

    int curr = (end+1)%N, idx2 = (end+1)%N, cnt=0;
    while (cnt < N) {
        int val = parent2[idx2];
        bool present = false;
        for (int k=start; k<=end; ++k)
            if (child[k]==val) { present=true; break; }
        if (!present) {
            child[curr] = val;
            curr = (curr+1)%N;
        }
        idx2 = (idx2+1)%N;
        cnt++;
    }
}

void initialize_population(int* orders, int pop_size, int N, std::mt19937& rng) {
    std::vector<int> tmp(N);
    for (int i = 0; i < N; ++i) tmp[i]=i;
    for (int i = 0; i < pop_size; ++i) {
        std::shuffle(tmp.begin(), tmp.end(), rng);
        for (int j = 0; j < N; ++j)
            orders[i*N+j] = tmp[j];
    }
}

void elitism(const int* orders, const int* fitness, int* elite_orders, int& elite_fitness, int pop_size, int N) {
    int best = 0;
    for (int i=1;i<pop_size;++i)
        if (fitness[i]<fitness[best])
            best=i;
    for (int j=0;j<N;++j)
        elite_orders[j] = orders[best*N+j];
    elite_fitness = fitness[best];
}

void runTest(int N, const std::vector<std::vector<CostRange>>& matrixCPU) {
    int pop_size = 100;

    CostRange* d_matrix;
    gpuErrchk(cudaMallocManaged(&d_matrix, N*N*sizeof(CostRange)));
    flattenMatrix(matrixCPU, d_matrix, N);

    int* d_orders;
    int* d_fitness;
    gpuErrchk(cudaMallocManaged(&d_orders, pop_size*N*sizeof(int)));
    gpuErrchk(cudaMallocManaged(&d_fitness, pop_size*sizeof(int)));

    curandState* d_state;
    gpuErrchk(cudaMalloc(&d_state, pop_size*sizeof(curandState)));

    std::random_device rd;
    std::mt19937 rng(rd());

    initialize_population(d_orders, pop_size, N, rng);

    setup_kernel<<<(pop_size+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_state, time(0), pop_size);
    cudaDeviceSynchronize();

    auto start = std::chrono::high_resolution_clock::now();
    for (int gen=0; gen<MAX_GENERATIONS; ++gen) {
        evaluate_population_kernel<<<(pop_size+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(
            d_orders, d_fitness, d_matrix, N, pop_size, d_state);
        cudaDeviceSynchronize();

        std::vector<int> elite_order(N);
        int elite_fitness;
        elitism(d_orders, d_fitness, elite_order.data(), elite_fitness, pop_size, N);

        std::vector<int> new_orders(pop_size*N);

        for (int j=0;j<N;++j)
            new_orders[j]=elite_order[j];

        for (int i=ELITISM;i<pop_size;++i) {
            std::vector<int> parent1(N), parent2(N), child(N);

            tournament_selection(d_orders, d_fitness, pop_size, N, rng, parent1.data());
            tournament_selection(d_orders, d_fitness, pop_size, N, rng, parent2.data());
            ox_crossover(parent1.data(), parent2.data(), child.data(), N, rng);

            for (int j=0;j<N;++j)
                new_orders[i*N+j]=child[j];
        }

        std::copy(new_orders.begin(), new_orders.end(), d_orders);

        mutate_kernel<<<(pop_size+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(
            d_orders, N, pop_size, MUTATION_RATE, d_state);
        cudaDeviceSynchronize();
    }
    evaluate_population_kernel<<<(pop_size+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(
            d_orders, d_fitness, d_matrix, N, pop_size, d_state);
    cudaDeviceSynchronize();

    int best_idx = 0;
    for (int i=1;i<pop_size;++i)
        if (d_fitness[i]<d_fitness[best_idx]) best_idx=i;

    auto stop = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double>(stop - start).count();

    std::cout << "N=" << N << "  Best cost: " << d_fitness[best_idx] << "   Time: " << ms << "s" << std::endl;

    cudaFree(d_matrix);
    cudaFree(d_orders);
    cudaFree(d_fitness);
    cudaFree(d_state);
}

int main(int argc, char* argv[]) {
    const int MAX_N = 5000;
    auto matrix = readMatrixFromFile("matrix.txt", MAX_N);
    if (matrix.empty() || (int)matrix.size() < MAX_N || (int)matrix[0].size() < 1000) {
        std::cerr << "Error loading matrix!" << std::endl;
        return 1;
    }
    if (argc > 1 && std::string(argv[1]) == "test") {
        std::vector<int> city_sizes = {500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000};
        for (int size : city_sizes) {
            runTest(size, matrix);
        }
    } else {
        runTest(1000, matrix);
    }
    return 0;
}
