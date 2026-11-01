#include <iostream>
#include <random>
#include <vector>
#include <cstddef>
#include <array>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <iomanip>


struct RunInfo;
std::vector<int> make_random_lattice(std::size_t L, std::mt19937& rng);
void print_lattice(const std::vector<int>& spins, std::size_t L);
std::size_t idx(int row, int col, int L);                             // Outputs the index of an agent in the lattice with its row and column positions
std::array<std::size_t, 4> neighbours(std::size_t i, std::size_t L); // Returns array containing positions of neighbours for the ith agent in a lattice of length L
const double local_field(const std::vector<int>& spins, std::size_t i, std::size_t L, double J, double h); // returns h_i (field experienced by ith agent) for a given coupling strength (J) and external field h
const double prob_up(double h_i, double beta);
void heat_bath_update(std::vector<int>& spins, std::size_t i, std::size_t L, double J, double h, double beta, std::mt19937& rng);
void sweep(std::vector<int>& spins, std::size_t L, double J, double h, double beta, std::mt19937& rng);
const double magnetisation(const std::vector<int>& spins);
std::vector<double> run_magnetisation(std::vector<int>& spins, std::size_t L, double J, double h, double beta, std::size_t n_sweeps, std::mt19937& rng);
void burn_in(std::vector<int>& spins, std::size_t L, double J, double h, double beta, std::size_t n_burns, std::mt19937& rng);
double mean(const std::vector<double>& series);
double standard_deviation(const std::vector<double>& series, double mean);
std::vector<double> differences(const std::vector<double>& m);
std::vector<std::size_t> extract_events(const std::vector<double>& differences, double theta);
bool write_events(const std::string& path, const std::vector<std::size_t>& events, const RunInfo& info);


struct RunInfo
    {
        std::size_t L{};
        double J{};
        double h{};
        double T{};
        double beta{};
        double theta{};
        unsigned int seed{};
        std::size_t n_burns{};
        std::size_t n_sweeps{};
    } ;

int main()
{

 
    // Our lattice values and sources of randomness

    const std::size_t L {100};      // Length of our lattice
    const int seed{11037};         // Seed to ensure results are reproduced each run
    std::mt19937 rng{seed};


    // Parameters for our model
    const double J {0.0};
    const double h {0.0};
    const double T {6};
    const double beta {1.0 / T};   // taking k_b = 1 : I am working in natural units


    // How many times to sweep and burn in
    const int n_burns{10000};     // Does n_burns sweeps without tracking magnetisations, giving system time to equilibriate
    const int n_sweeps{10000};


    std::cout << "Seed: " << seed << '\n';
    std::cout << "Parameters: \n" 
              << "J: " << J << '\n'
              << "h: " << h << '\n'
              << "T: " << T << '\n'
              << "Beta: " << beta<< '\n';

    // Initial Spins of lattice: 
    //std::vector<int> spins{make_random_lattice(L, rng)};     // Random Start

    std::vector<int> spins(L*L, 1);                           // All positive

    //std::vector<int> spins(L*L, -1);                       // All Negative
    
    burn_in(spins, L, J, h, beta, n_burns, rng);

    std::vector<double> m_series {run_magnetisation(spins, L, J, h, beta, n_sweeps, rng)}; // Vector of magnetisations for each sweep

    double delta_std {standard_deviation(differences(m_series), mean(differences(m_series)))};


    double multiplier{2.5};
    double theta{ multiplier * delta_std};

    const RunInfo info{L, J, h, T, beta, theta, seed, n_burns, n_sweeps};
    
    auto events{extract_events(differences(m_series), theta)};

    write_events("J= 0_2.txt", events, info);


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

const double magnetisation(const std::vector<int>& spins)
{
    double spin_sum {std::accumulate(spins.begin(), spins.end(), 0.0)};

    return spin_sum / spins.size();
}

std::vector<double> run_magnetisation(std::vector<int>& spins, std::size_t L, double J, double h, double beta, std::size_t n_sweeps, std::mt19937& rng)
{
    std::vector<double> magnetisations{};

    for (std::size_t i{}; i < n_sweeps; ++i)
    {
        sweep(spins,L, J, h, beta, rng);
        magnetisations.push_back(magnetisation(spins)); // Not including magnetisation of intial state.
    }

    return magnetisations;
}

void burn_in(std::vector<int>& spins, std::size_t L, double J, double h, double beta, std::size_t n_burns, std::mt19937& rng)
{
    for( std::size_t i{}; i < n_burns; ++i)
    {
        sweep(spins, L, J, h, beta, rng);
    }
}

double mean(const std::vector<double>& series)  
{
    double sum {std::accumulate(series.begin(), series.end(), 0.0)};

    return sum / series.size();
}

double standard_deviation(const std::vector<double>& series, double mean)
{
    double sum{};
    for (double value : series)
    {
        sum += (std::pow((value - mean), 2));
    }

    return std::sqrt(sum / (series.size() - 1)); // n - 1 as these sweeps are only a sample out of all the possible runs
}

std::vector<double> differences(const std::vector<double>& m_series)
{
    std::vector<double> d{};

    if (m_series.size() < 2)
    {
        return d;
    }

    for (std::size_t i{1}; i < m_series.size(); ++i)
    {
        d.push_back(m_series[i] - m_series[i - 1]);
    }
    return d;
}

std::vector<std::size_t> extract_events(const std::vector<double>& differences, double theta)
{

    std::vector<std::size_t> events{};

    for(std::size_t i{}; i < differences.size(); ++i)
    {
        if (std::abs(differences[i]) > theta)
        {
            events.push_back(i + 1);  // i + 1 as we want the index in the m_series vector that corresponds to the value after the change
        }
    }

    return events;
}

bool write_events(const std::string& path, const std::vector<std::size_t>& events, const RunInfo& info)
{

    std::ofstream file {path};

    if (!file)
    {
        std::cerr << " Couldn't open file. \n";
        return false;
    }
    
    file << std::setprecision(8);
    file << "# L= " << info.L << " |J= " << info.J << " |h= " << info.h << " |T= " << info.T 
         << " |Beta= " << info.beta << " |Theta= " << info.theta << " |Seed= " << info.seed 
         << " |Number of burns= " << info.n_burns << " |Number of sweeps= " << info.n_sweeps << '\n'
         << "|Window= " << info.n_sweeps << '\n';


    for(std::size_t event : events)
    {
        file << event << '\n';
    }

   return true;
}