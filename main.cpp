#include <iostream>
#include <random>
#include <vector>
#include <cstddef>
#include <array>
#include <cmath>



std::vector<int> make_random_lattice(std::size_t L, std::mt19937& rng);
void print_lattice(const std::vector<int>& spins, std::size_t L);
std::size_t idx(int row, int col, int L);                             // Outputs the index of an agent in the lattice with its row and column positions
std::array<std::size_t, 4> neighbours(std::size_t i, std::size_t L); // Returns array containing positions of neighbours for the ith agent in a lattice of length L
double local_field(const std::vector<int>& spins, std::size_t i, std::size_t L, double J, double h); // returns h_i (field experienced by ith agent) for a given coupling strength (J) and external field h
void heat_bath_update(std::vector<int>& spins, std::size_t i, std::size_t L, double J, double h, double beta, std::mt19937 rng);


int main()
{

    const std::size_t L {10};       // Length of our lattice
    const int seed{11037};         // Seed to ensure results are reproduced each run

    std::mt19937 rng{seed};

    std::vector<int> spins{make_random_lattice(L, rng)};     // Produces a vector of N spins

    print_lattice(spins, L);

}

std::vector<int> make_random_lattice(std::size_t L, std::mt19937& rng)
{
    const std::size_t N { L * L}; // Number of agents in lattice of length L
    std::vector<int> spins(N);    // Vector of N spins (each corresponding to the spin of a specific agent)

    std::uniform_int_distribution<int> coin{0,1};  // Using an infinite temperature start. This makes the probability of being in either spin state equal 

    for (std::size_t i{}; i < N; ++i)
    {
        spins[i] = coin(rng) == 1 ? 1 : -1;   // For Loop sets the spin of agent i as +1 or -1, depending on coin flip outcome
    }

    return spins;

}

void print_lattice(const std::vector<int>& spins, std::size_t L)
{
    for (std::size_t row{}; row < L; ++row)
    {
        for (std::size_t col{}; col < L; ++col)
        {
            std::cout << (spins[idx(row, col, L)] == 1 ? '+' : '-') << ' ';  // row * L + col treats 2D grid as 1D storage for accessing specific agents
        }
        std::cout << '\n';
    }

}

std::size_t idx(int row, int col, int L)  
{
    return row * L + col;
}

std::array<std::size_t, 4> neighbours(std::size_t i, std::size_t L)
{

    // Inverses of row * L + col to get row and col
    const std::size_t row{ i / L};  
    const std::size_t col {i % L};

    const std::size_t up {idx((row + L - 1) % L, col, L)};  // Adding L when substracting to ensure unsigned values don't wrap around to top of range
    const std::size_t down{idx((row + 1) % L, col , L)};
    const std::size_t left{idx(row, (col + L - 1) % L, L)};
    const std::size_t right{idx(row, (col + 1) % L, L)};

    return {up, down, left, right};

}

double local_field(const std::vector<int>& spins, std::size_t i, std::size_t L, double J, double h)
{
    int neighbour_sum{};

    for (std::size_t j : neighbours(i, L)) // Iterates through neighbours of ith agent in lattice of length L
    {
        neighbour_sum += spins[j];
    }

    return J * neighbour_sum + h;  // Local field experienced by ith agent (h_i), h represents external field on system
}

double prob_up(double h_i, double beta)
{
    return 1.0/ (1.0 + std::exp(-2.0 * beta * h_i));
}

void heat_bath_update(std::vector<int>& spins, std::size_t i, std::size_t L, double J, double h, double beta, std::mt19937& rng)
{
    double local{local_field(spins, i, L, J, h)};

    double p_up { prob_up(local, beta)};
    std::uniform_real_distribution<double> u {0, 1}; // Pick a random number between 0 and 1 (exclusive)
    
     if (u(rng) < p_up)
    {
            spins[i] = 1;
    }
    else
    {
            spins[i] = -1;
    }
    
}