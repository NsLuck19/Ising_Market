#include <iostream>
#include <random>
#include <vector>
#include <cstddef>

std::vector<int> make_random_lattice(std::size_t L, std::mt19937& rng);

int main()
{

    const std::size_t L {8};       // Length of our lattice
    const int seed{11037};         // Seed to ensure results are reproduced each run

    std::mt19937 rng{seed};

    std::vector<int> spins{make_random_lattice(L, rng)};     // Produces a vector of N zeroes

    for (std::size_t row{}; row < L; ++row)
    {
        for (std::size_t col{}; col < L; ++col)
        {
            std::cout << (spins[row * L + col] == 1 ? '+' : '-') << ' ';  // row * L + col treats 2D grid as 1D storage for accessing specific agents
        }
        std::cout << '\n';
    }

}

std::vector<int> make_random_lattice(std::size_t L, std::mt19937& rng)
{
    const std::size_t N { L * L}; // Number of agents in lattice of length L
    std::vector<int> spins(N);    // Vector of N spins (each corresponing to the spin of a specific agent)

    std::uniform_int_distribution<int> coin{0,1};  // Using an infinite temperature start. This makes the probability of being in either spin state equal 

    for (std::size_t i{}; i < N; ++i)
    {
        spins[i] = coin(rng) == 1 ? 1 : -1;   // For Loop sets the spin of agent i as +1 or -1, depending on coin flip outcome
    }

    return spins;


}