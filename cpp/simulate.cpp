#include "kv_pool.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <thread>

struct Request { std::size_t prompt, decode; };
struct Active { std::size_t id, length, remaining; };
struct Result {
    std::string name;
    std::size_t peak_capacity = 0, peak_logical = 0, peak_active = 0, ticks = 0;
    std::size_t allocations = 0, frees = 0, cow = 0, copied = 0;
    double capacity_sum = 0, occupied_sum = 0, logical_sum = 0, elapsed_ms = 0;
};

Result run(const std::vector<Request>& requests, std::size_t block_size,
           std::size_t prefix_length, std::size_t max_seq, std::size_t arrivals,
           std::size_t pool_blocks, const std::string& mode) {
    kv::BlockPool pool(pool_blocks, block_size);
    const bool paged = mode != "contiguous_reserved";
    const bool sharing = mode == "paged_shared_prefix";
    std::vector<kv::Token> prefix(prefix_length);
    for (std::size_t i = 0; i < prefix.size(); ++i) prefix[i] = i;
    std::size_t prefix_id = 0;
    auto started = std::chrono::steady_clock::now();
    if (sharing) { prefix_id = pool.create(); pool.append(prefix_id, prefix); }
    std::vector<Active> active;
    std::size_t next = 0;
    Result result;
    result.name = mode;
    while (next < requests.size() || !active.empty()) {
        for (std::size_t i = 0; i < arrivals && next < requests.size(); ++i, ++next) {
            const auto& request = requests[next];
            auto id = paged ? (sharing ? pool.fork_prefix(prefix_id, prefix_length) : pool.create()) : next;
            if (paged) {
                std::vector<kv::Token> prompt(request.prompt, 1000 + next);
                std::copy(prefix.begin(), prefix.end(), prompt.begin());
                if (sharing) prompt.erase(prompt.begin(), prompt.begin() + prefix_length);
                pool.append(id, prompt);
            } else ++result.allocations;
            active.push_back({id, request.prompt, request.decode});
        }
        // Observe after prefill/admission and before this tick's decode/release.
        std::size_t logical = 0;
        for (const auto& request : active) logical += request.length;
        const auto capacity = paged ? pool.allocated_token_capacity() : active.size() * max_seq;
        const auto occupied = paged ? pool.used_token_slots() : logical;
        result.peak_capacity = std::max(result.peak_capacity, capacity);
        result.peak_logical = std::max(result.peak_logical, logical);
        result.peak_active = std::max(result.peak_active, active.size());
        result.capacity_sum += capacity;
        result.occupied_sum += occupied;
        result.logical_sum += logical;
        ++result.ticks;
        for (auto& request : active) {
            if (paged) pool.append(request.id, {static_cast<kv::Token>(request.length)});
            ++request.length;
            --request.remaining;
        }
        auto it = active.begin();
        while (it != active.end()) {
            if (!it->remaining) {
                if (paged) pool.release(it->id); else ++result.frees;
                it = active.erase(it);
            } else ++it;
        }
    }
    if (sharing) pool.release(prefix_id);
    pool.validate();
    if (pool.allocated_blocks()) throw std::logic_error("leaked physical blocks");
    result.elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    if (paged) {
        auto counters = pool.counters();
        result.allocations = counters.allocations;
        result.frees = counters.frees;
        result.cow = counters.copy_on_write;
        result.copied = counters.copied_token_slots;
        // Allocations after a tick's observation can transiently increase the true peak.
        result.peak_capacity = counters.high_water_blocks * block_size;
    }
    return result;
}

int main(int argc, char** argv) {
    try {
        std::size_t seed = 42, count = 256, block_size = 16, prefix = 30;
        std::size_t max_seq = 128, arrivals = 3, pool_blocks = 4096;
        for (int i = 1; i < argc; i += 2) {
            if (i + 1 == argc) throw std::invalid_argument("flags need integer values");
            auto value = std::stoull(argv[i + 1]);
            std::string flag = argv[i];
            if (flag == "--seed") seed = value;
            else if (flag == "--requests") count = value;
            else if (flag == "--block-size") block_size = value;
            else if (flag == "--prefix") prefix = value;
            else if (flag == "--max-seq") max_seq = value;
            else if (flag == "--arrival-rate") arrivals = value;
            else if (flag == "--pool-blocks") pool_blocks = value;
            else throw std::invalid_argument("unknown option: " + flag);
        }
        if (!count || !block_size || !arrivals || !pool_blocks || max_seq < prefix + 96)
            throw std::invalid_argument("positive dimensions and max-seq >= prefix + 96 required");
        std::mt19937 generator(seed);
        std::uniform_int_distribution<std::size_t> suffix(0, 64), decode(8, 32);
        std::vector<Request> requests;
        for (std::size_t i = 0; i < count; ++i) requests.push_back({prefix + suffix(generator), decode(generator)});
        std::vector<Result> results;
        for (auto mode : {"contiguous_reserved", "paged_private", "paged_shared_prefix"})
            results.push_back(run(requests, block_size, prefix, max_seq, arrivals, pool_blocks, mode));
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "{\n  \"schema_version\": 1,\n  \"kind\": \"cpu_control_plane_simulation\",\n";
        std::cout << "  \"environment\": {\"compiler\": \"" << __VERSION__
                  << "\", \"hardware_threads\": " << std::thread::hardware_concurrency() << "},\n";
        std::cout << "  \"config\": {\"seed\": " << seed << ", \"requests\": " << count
                  << ", \"block_size\": " << block_size << ", \"common_prefix_tokens\": " << prefix
                  << ", \"max_sequence_tokens\": " << max_seq << ", \"arrivals_per_tick\": " << arrivals
                  << ", \"pool_blocks\": " << pool_blocks << "},\n";
        std::cout << "  \"measurement_note\": \"Capacity is assigned token slots, not pool preallocation or measured process RAM. Timings include bookkeeping and occupancy scans; no GPU/inference speed claim.\",\n";
        std::cout << "  \"results\": [\n";
        for (std::size_t i = 0; i < results.size(); ++i) {
            const auto& r = results[i];
            std::cout << "    {\"mode\": \"" << r.name << "\", \"ticks\": " << r.ticks
                      << ", \"peak_active_requests\": " << r.peak_active
                      << ", \"peak_allocated_token_capacity\": " << r.peak_capacity
                      << ", \"peak_request_logical_tokens\": " << r.peak_logical
                      << ", \"mean_allocated_token_capacity\": " << r.capacity_sum / r.ticks
                      << ", \"physical_occupancy_ratio\": " << r.occupied_sum / r.capacity_sum
                      << ", \"request_logical_tokens_per_allocated_slot\": " << r.logical_sum / r.capacity_sum
                      << ", \"mean_capacity_reduction_vs_reserved\": " << 1 - r.capacity_sum / results[0].capacity_sum
                      << ", \"block_or_reservation_allocations\": " << r.allocations
                      << ", \"block_or_reservation_frees\": " << r.frees
                      << ", \"copy_on_write_operations\": " << r.cow
                      << ", \"copied_token_markers\": " << r.copied
                      << ", \"simulation_elapsed_ms\": " << r.elapsed_ms << "}"
                      << (i + 1 < results.size() ? "," : "") << "\n";
        }
        std::cout << "  ]\n}\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
