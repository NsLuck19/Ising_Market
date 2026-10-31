#include <iostream>
#include <random>
#include <vector>
#include <cstddef>
#include <array>
#include <cmath>
#include <algorithm>



std::vector<int> make_random_lattice(std::size_t L, std::mt19937& rng);
void print_lattice(const std::vector<int>& spins, std::size_t L);
std::size_t idx(int row, int col, int L);                             // Outputs the index of an agent in the lattice with its row and column positions
std::array<std::size_t, 4> neighbours(std::size_t i, std::size_t L); // Returns array containing positions of neighbours for the ith agent in a lattice of length L
const double local_field(const std::vector<int>& spins, std::size_t i, std::size_t L, double J, double h); // returns h_i (field experienced by ith agent) for a given coupling strength (J) and external field h
const double prob_up(double h_i, double beta);
void heat_bath_update(std::vector<int>& spins, std::size_t i, std::size_t L, double J, double h, double beta, std::mt19937& rng);
void sweep(std::vector<int>& spins, std::size_t L, double J, double h, double beta, std::mt19937& rng);
double magnetisation(std::vector<int>& spins);
std::vector<double> run_magnetisation(std::vector<int>& spins, std::size_t L, double J, double h, double beta, std::size_t n_sweeps, std::mt19937& rng);


int main()
{
 
    // Our lattice values and sources of randomness

    const std::size_t L {100};      // Length of our lattice
    const int seed{11037};         // Seed to ensure results are reproduced each run
    std::mt19937 rng{seed};


    // Parameters for our model
    const double J {1.0};
    const double h {0.0};
    const double T {1.5};
    const double beta {1.0 / T};   // taking k_b = 1 : I am working in natural units


    // How many times to sweep
    const int n_sweeps{1000};


    // Initial Spin vector if we want a random start
    std::vector<int> spins{make_random_lattice(L, rng)};     // Produces a vector of N spins

    //Initial Spin vector if we want a lattice of +1s
    //std::vector<int> spins(L*L, 1);

    //Initial Spin vector if we wwant a lattice of -1s
    //std::vector<int> spins(L*L, -1);

   // std::vector<double> temp_vect {5.0, 3.0, 2.0, 1.5};  // Temperature vector for doing tests for a range of temperatures
    
    //print_lattice(spins, L);

    std::vector<double> magnetisations {run_magnetisation(spins, L, J, h, beta, n_sweeps, rng)};

    double sum{};
    double std_sum{};
    for (std::size_t i{}; i < magnetisations.size(); ++i)
    {
        std::cout << magnetisations[i] << '\n';
        sum += magnetisations[i];
    }

    double mean{ sum / magnetisations.size()};

    for (std::size_t i {}; i < magnetisations.size(); ++i)
    {
        std_sum += std::pow((magnetisations[i] - mean), 2);
    }

    double standard_deviation{ std::pow((std_sum/magnetisations.size()), 0.5)};

    std::cout << "Mean: " << mean << '\n' 
              << "Standard deviation: " << standard_deviation << '\n';




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

const double local_field(const std::vector<int>& spins, std::size_t i, std::size_t L, double J, double h)
{
    int neighbour_sum{};

    for (std::size_t j : neighbours(i, L)) // Iterates through neighbours of ith agent in lattice of length L
    {
        neighbour_sum += spins[j];
    }

    return J * neighbour_sum + h;  // Local field experienced by ith agent (h_i), h represents external field on system
}

const double prob_up(double h_i, double beta)
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

void sweep(std::vector<int>& spins, std::size_t L, double J, double h, double beta, std::mt19937& rng)
{
    const std::size_t N {L*L};
    std::uniform_int_distribution<std::size_t> agent_choice {0, N-1};

    for (std::size_t i {}; i < N; ++i)
    {
        std::size_t choice {agent_choice(rng)};  // Pick a random agent in the lattice
        heat_bath_update(spins, choice, L, J, h, beta, rng);
    }
}

double magnetisation(std::vector<int>& spins)
{
    double spin_sum {};

    for (int spin : spins)
    {
        spin_sum += spin;
    }
    return spin_sum / spins.size();

}

std::vector<double> run_magnetisation(std::vector<int>& spins, std::size_t L, double J, double h, double beta, std::size_t n_sweeps, std::mt19937& rng)
{

    std::vector<int> lattice {spins};
    std::vector<double> magnetisations{};

    for (std::size_t i{}; i < n_sweeps; ++i)
    {
        sweep(lattice,L, J, h, beta, rng);
        magnetisations.push_back(magnetisation(lattice)); // Not including magnetisation of intial state.
    }

    return magnetisations;
}