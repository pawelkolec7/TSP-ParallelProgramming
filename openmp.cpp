#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <chrono>
#include <omp.h>

#define MAX_GENERATIONS 10
#define MUTATION_RATE 0.05
#define ELITISM 1

const std::string FILENAME = "matrix.txt";

struct CostRange {
    int min_cost;
    int max_cost;
};

struct Chromosome {
    std::vector<int> order;
    int fitness;
};

using Matrix = std::vector<std::vector<CostRange>>;
using Population = std::vector<Chromosome>;

Matrix readMatrixFromFile(const std::string& filename, int N) {
    Matrix matrix(N, std::vector<CostRange>(N));
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

int random_cost(const CostRange& cr, std::mt19937& rng) {
    if (cr.min_cost == cr.max_cost) return cr.min_cost;
    std::uniform_int_distribution<int> dist(cr.min_cost, cr.max_cost);
    return dist(rng);
}

int compute_fitness(const Chromosome& c, const Matrix& m, std::mt19937& rng) {
    int sum = 0;
    int n = c.order.size();
    for (int i = 0; i < n; ++i) {
        int from = c.order[i];
        int to = c.order[(i + 1) % n];
        sum += random_cost(m[from][to], rng);
    }
    return sum;
}

void initialize_population(Population& pop, int pop_size, int n, std::mt19937& rng) {
    pop.resize(pop_size);
    for (int i = 0; i < pop_size; ++i) {
        pop[i].order.resize(n);
        for (int j = 0; j < n; ++j) pop[i].order[j] = j;
        std::shuffle(pop[i].order.begin(), pop[i].order.end(), rng);
        pop[i].fitness = 0;
    }
}

void evaluate(Population& pop, const Matrix& matrix) {
    #pragma omp parallel
    {
        std::mt19937 rng((unsigned int)(std::random_device{}() + omp_get_thread_num()*10000));
        #pragma omp for
        for (int i = 0; i < (int)pop.size(); ++i) {
            pop[i].fitness = compute_fitness(pop[i], matrix, rng);
        }
    }
}

Chromosome tournament_selection(const Population& pop, std::mt19937& rng) {
    std::uniform_int_distribution<int> dist(0, pop.size() - 1);
    const Chromosome& a = pop[dist(rng)];
    const Chromosome& b = pop[dist(rng)];
    return (a.fitness < b.fitness) ? a : b;
}

void crossover(const Chromosome& parent1, const Chromosome& parent2, Chromosome& child, std::mt19937& rng) {
    int n = parent1.order.size();
    std::uniform_int_distribution<int> dist(0, n - 1);
    int start = dist(rng);
    int end = dist(rng);
    if (start > end) std::swap(start, end);

    child.order.assign(n, -1);
    for (int i = start; i <= end; ++i) {
        child.order[i] = parent1.order[i];
    }
    int current = 0;
    for (int i = 0; i < n; ++i) {
        int val = parent2.order[i];
        if (std::find(child.order.begin() + start, child.order.begin() + end + 1, val) == child.order.begin() + end + 1) {
            while (child.order[current] != -1) ++current;
            child.order[current] = val;
        }
    }
}

void mutate(Chromosome& c, std::mt19937& rng) {
    std::uniform_real_distribution<double> rate_dist(0.0, 1.0);
    std::uniform_int_distribution<int> idx_dist(0, c.order.size() - 1);
    for (int i = 0; i < (int)c.order.size(); ++i) {
        if (rate_dist(rng) < MUTATION_RATE) {
            int j = idx_dist(rng);
            std::swap(c.order[i], c.order[j]);
        }
    }
}

void sort_population(Population& pop) {
    std::sort(pop.begin(), pop.end(), [](const Chromosome& a, const Chromosome& b) {
        return a.fitness < b.fitness;
    });
}

void runTest(int N, const Matrix& fullMatrix) {
    std::mt19937 rng(std::random_device{}());
    int pop_size = 100;
    Matrix matrix(N, std::vector<CostRange>(N));
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            matrix[i][j] = fullMatrix[i][j];

    Population pop;
    initialize_population(pop, pop_size, N, rng);

    auto start = std::chrono::high_resolution_clock::now();

    for (int gen = 0; gen < MAX_GENERATIONS; ++gen) {
        evaluate(pop, matrix);
        sort_population(pop);

        Population new_pop;
        for (int i = 0; i < ELITISM; ++i)
            new_pop.push_back(pop[i]);

        #pragma omp parallel
        {
            std::mt19937 thread_rng((unsigned int)(std::random_device{}() + omp_get_thread_num()*10000));
            #pragma omp for schedule(static)
            for (int i = ELITISM; i < pop_size; ++i) {
                Chromosome parent1 = tournament_selection(pop, thread_rng);
                Chromosome parent2 = tournament_selection(pop, thread_rng);

                Chromosome child;
                crossover(parent1, parent2, child, thread_rng);
                mutate(child, thread_rng);
                child.fitness = compute_fitness(child, matrix, thread_rng);

                #pragma omp critical
                new_pop.push_back(child);
            }
        }
        pop = std::move(new_pop);
    }

    auto stop = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double>(stop - start).count();

    sort_population(pop);
    std::cout << "N=" << N << "  Best cost: " << pop[0].fitness << "   Time: " << ms << "s" << std::endl;
}

int main(int argc, char* argv[]) {
    auto fullMatrix = readMatrixFromFile(FILENAME, 5000);

    if (fullMatrix.empty() || (int)fullMatrix.size() < 5000 || (int)fullMatrix[0].size() < 1000) {
        std::cerr << "Error loading matrix!" << std::endl;
        return 1;
    }

    if (argc > 1 && std::string(argv[1]) == "test") {
        std::vector<int> city_sizes = {500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000};
        for (int size : city_sizes) {
            runTest(size, fullMatrix);
        }
    } else {
        runTest(1000, fullMatrix);
    }

    return 0;
}
