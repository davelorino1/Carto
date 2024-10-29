#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <map>

// Structure to hold store pairs and their difference
struct store_pair {
    int test_store;
    int control_store;
    double abs_perc_diff;
};

// Function to read the CSV file and store the data in a vector
std::vector<store_pair> read_csv(const std::string& filename) {
    std::ifstream file(filename.c_str());
    std::string line, word;
    std::vector<store_pair> store_pairs;

    // Skip the header row
    if (std::getline(file, line)) {}

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        store_pair sp;

        std::getline(ss, word, ','); // Ignore the study_id

        std::getline(ss, word, ',');
        sp.test_store = std::atoi(word.c_str());

        std::getline(ss, word, ',');
        sp.control_store = std::atoi(word.c_str());

        std::getline(ss, word, ',');
        sp.abs_perc_diff = std::atof(word.c_str());
        //std::cout << "Reading Store Pair: " << sp.test_store << " with Control Store " << sp.control_store << " (Diff: " << sp.abs_perc_diff << ")\n";
        store_pairs.push_back(sp);
    }
    return store_pairs;
}

// Traditional comparison function for sorting by test_store and then by abs_perc_diff
bool compare_store_pairs(const store_pair& a, const store_pair& b) {
    if (a.test_store == b.test_store) {
        return a.abs_perc_diff < b.abs_perc_diff;
    }
    return a.test_store < b.test_store;
}

// Traditional comparison function for sorting by abs_perc_diff globally
bool compare_by_abs_perc_diff(const store_pair& a, const store_pair& b) {
    return a.abs_perc_diff < b.abs_perc_diff;
}

// Function to perform greedy matching with no reuse (each control store can be used only once)
std::vector<store_pair> greedy_matching(std::vector<store_pair>& store_pairs) {
    std::vector<store_pair> matched_pairs;
    std::map<int, bool> used_test_stores;
    std::map<int, bool> used_control_stores;

    // Sort the pairs by test_store and then by abs_perc_diff for easier matching
    // std::sort(store_pairs.begin(), store_pairs.end(), compare_store_pairs);

    for (std::vector<store_pair>::iterator sp = store_pairs.begin(); sp != store_pairs.end(); ++sp) {
        if (!used_test_stores[sp->test_store] && !used_control_stores[sp->control_store]) {
            matched_pairs.push_back(*sp);
            used_test_stores[sp->test_store] = true;
            used_control_stores[sp->control_store] = true;
            std::cout << "Greedy: Matched Test Store " << sp->test_store << " with Control Store " << sp->control_store << " (Diff: " << sp->abs_perc_diff << ")\n";
        }
    }

    return matched_pairs;
}

// Function to perform global optimized matching with no reuse (each control store can be used only once)
std::vector<store_pair> global_matching(std::vector<store_pair>& store_pairs) {
    std::vector<store_pair> matched_pairs;
    std::map<int, bool> used_test_stores;
    std::map<int, bool> used_control_stores;

    // Sort the pairs globally by abs_perc_diff to minimize the overall sum
    std::sort(store_pairs.begin(), store_pairs.end(), compare_by_abs_perc_diff);

    for (std::vector<store_pair>::iterator sp = store_pairs.begin(); sp != store_pairs.end(); ++sp) {
        if (!used_test_stores[sp->test_store] && !used_control_stores[sp->control_store]) {
            matched_pairs.push_back(*sp);
            used_test_stores[sp->test_store] = true;
            used_control_stores[sp->control_store] = true;
            
            //std::cout << "Global: Matched Test Store " << sp->test_store << " with Control Store " << sp->control_store << " (Diff: " << sp->abs_perc_diff << ")\n";
        }
    }

    return matched_pairs;
}

// Function to calculate the total absolute percentage difference
double calculate_total_difference(const std::vector<store_pair>& matched_pairs) {
    double total_difference = 0.0;
    for (std::vector<store_pair>::const_iterator pair = matched_pairs.begin(); pair != matched_pairs.end(); ++pair) {
        total_difference += pair->abs_perc_diff;
    }
    return total_difference;
}

// Function to write matched pairs to a CSV file
void write_csv(const std::string& filename, const std::vector<store_pair>& matched_pairs) {
    std::ofstream file(filename.c_str());
    file << "test_store,control_store,abs_perc_diff\n";
    for (std::vector<store_pair>::const_iterator pair = matched_pairs.begin(); pair != matched_pairs.end(); ++pair) {
        file << pair->test_store << ","
             << pair->control_store << ","
             << pair->abs_perc_diff << "\n";
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <csv_file_path>" << std::endl;
        return 1;
    }

    std::string filename = argv[1];  // File path provided as a command-line argument
    std::vector<store_pair> store_pairs = read_csv(filename);

    // Apply greedy matching with reuse limit of zero
    std::vector<store_pair> greedy_result = greedy_matching(store_pairs);
    double greedy_total_diff = calculate_total_difference(greedy_result);
    write_csv("greedy_matching.csv", greedy_result);

    // Apply global optimization matching with reuse limit of zero
    std::vector<store_pair> global_result = global_matching(store_pairs);
    double global_total_diff = calculate_total_difference(global_result);
    write_csv("global_matching.csv", global_result);

    std::cout << "Greedy Matching Total Difference: " << greedy_total_diff << std::endl;
    std::cout << "Global Matching Total Difference: " << global_total_diff << std::endl;

    return 0;
}
