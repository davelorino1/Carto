#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <map>
#include <sys/stat.h> // For mkdir
#include <sys/types.h>
#include <unistd.h>   // For getcwd

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
        store_pairs.push_back(sp);
    }
    return store_pairs;
}

// Comparison functions
bool compare_store_pairs(const store_pair& a, const store_pair& b) {
    if (a.test_store == b.test_store) {
        return a.abs_perc_diff < b.abs_perc_diff;
    }
    return a.test_store < b.test_store;
}

bool compare_by_abs_perc_diff(const store_pair& a, const store_pair& b) {
    return a.abs_perc_diff < b.abs_perc_diff;
}

// Matching functions
std::vector<store_pair> greedy_matching(std::vector<store_pair>& store_pairs) {
    std::vector<store_pair> matched_pairs;
    std::map<int, bool> used_test_stores;
    std::map<int, bool> used_control_stores;

    for (std::vector<store_pair>::iterator sp = store_pairs.begin(); sp != store_pairs.end(); ++sp) {
        if (!used_test_stores[sp->test_store] && !used_control_stores[sp->control_store]) {
            matched_pairs.push_back(*sp);
            used_test_stores[sp->test_store] = true;
            used_control_stores[sp->control_store] = true;
        }
    }

    return matched_pairs;
}

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
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <csv_file_path> <campaign_id>" << std::endl;
        return 1;
    }

    std::string filename = argv[1];    // Input CSV file path
    std::string campaign_id = argv[2]; // Campaign ID

    // Create 'outputs' directory inside 'Match Maker' directory if it doesn't exist
    char cwd[1024];
    if (getcwd(cwd, sizeof(cwd)) == NULL) {
        perror("getcwd() error");
        return 1;
    }
    std::string current_dir(cwd);
    std::string outputs_dir = current_dir + "/Match Maker/outputs";

    struct stat info;
    if (stat(outputs_dir.c_str(), &info) != 0) {
        // Directory doesn't exist, create it
        if (mkdir(outputs_dir.c_str(), 0755) != 0) {
            perror("Failed to create outputs directory");
            return 1;
        }
    } else if (!(info.st_mode & S_IFDIR)) {
        std::cerr << outputs_dir << " exists but is not a directory!" << std::endl;
        return 1;
    }

    std::vector<store_pair> store_pairs = read_csv(filename);

    // Apply greedy matching with reuse limit of zero
    std::vector<store_pair> greedy_result = greedy_matching(store_pairs);
    double greedy_total_diff = calculate_total_difference(greedy_result);

    // Construct output filenames using campaign_id
    std::string greedy_output_file = outputs_dir + "/" + campaign_id + "_greedy_matching.csv";
    // write_csv(greedy_output_file, greedy_result);

    // Apply global optimization matching with reuse limit of zero
    std::vector<store_pair> global_result = global_matching(store_pairs);
    double global_total_diff = calculate_total_difference(global_result);

    std::string global_output_file = outputs_dir + "/" + campaign_id + "_global_matching.csv";
    write_csv(global_output_file, global_result);

    std::cout << "Greedy Matching Total Difference: " << greedy_total_diff << std::endl;
    std::cout << "Global Matching Total Difference: " << global_total_diff << std::endl;

    std::cout << "Output files generated in: " << outputs_dir << std::endl;

    return 0;
}
