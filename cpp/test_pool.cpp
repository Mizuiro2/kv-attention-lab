#include "kv_pool.hpp"
#include <iostream>
#include <random>

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

int main() {
    {
        kv::BlockPool pool(8, 4);
        auto parent = pool.create();
        pool.append(parent, {1, 2, 3, 4, 5, 6});
        auto child = pool.fork_prefix(parent, 5);
        require(pool.allocated_blocks() == 2, "prefix shares physical blocks");
        require(pool.references()[pool.sequence(parent).blocks[1]] == 2, "partial refcount");
        pool.append(child, {99, 100, 101, 102});
        require(pool.read(parent) == std::vector<kv::Token>({1, 2, 3, 4, 5, 6}), "COW preserves parent");
        require(pool.read(child) == std::vector<kv::Token>({1, 2, 3, 4, 5, 99, 100, 101, 102}), "COW child data");
        require(pool.counters().copy_on_write == 1, "one COW");
        require(pool.allocated_blocks() == 4, "two shared/unshared tables");
        require(pool.used_token_slots() == 11, "shared physical occupancy");
        pool.validate();
        pool.release(parent);
        pool.validate();
        require(pool.read(child).at(5) == 99, "child lives after parent release");
        pool.release(child);
        require(pool.free_blocks() == 8, "all blocks reclaimed");
        pool.validate();
    }
    {
        kv::BlockPool pool(2, 4);
        auto parent = pool.create();
        pool.append(parent, {1, 2, 3});
        auto child = pool.fork_prefix(parent, 3);
        auto refs = pool.references();
        auto table = pool.sequence(child).blocks;
        auto before = pool.read(child);
        bool failed = false;
        try { pool.append(child, {4, 5, 6, 7, 8, 9}); } catch (const std::bad_alloc&) { failed = true; }
        require(failed, "COW plus expansion must reject atomically");
        require(pool.references() == refs && pool.sequence(child).blocks == table, "OOM refcounts/table rollback");
        require(pool.read(child) == before && pool.read(parent) == before && pool.free_blocks() == 1, "OOM payload rollback");
        pool.validate();
        pool.append(child, {4});
        pool.validate();
        require(pool.read(parent) == before, "successful COW after OOM");
    }
    {
        kv::BlockPool pool(4, 4);
        auto source = pool.create();
        pool.append(source, {1, 2, 3, 4});
        auto full = pool.fork_prefix(source, 4);
        pool.append(full, {5});
        require(pool.counters().copy_on_write == 0, "full shared block needs no COW");
        auto empty = pool.fork_prefix(source, 0);
        pool.append(empty, {});
        pool.append(empty, {42});
        require(pool.read(empty) == std::vector<kv::Token>({42}), "empty prefix works");
        bool rejected = false;
        try { pool.fork_prefix(source, 5); } catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "invalid prefix rejects");
        pool.validate();
    }
    {
        // Differential randomized test against plain token vectors, including OOM.
        kv::BlockPool pool(32, 8);
        std::map<kv::SequenceId, std::vector<kv::Token>> oracle;
        std::mt19937 generator(2718);
        for (int step = 0; step < 2500; ++step) {
            if (oracle.empty() || generator() % 5 == 0) oracle[pool.create()] = {};
            auto it = oracle.begin();
            std::advance(it, generator() % oracle.size());
            auto operation = generator() % 3;
            if (operation == 0) {
                auto prefix = generator() % (it->second.size() + 1);
                auto id = pool.fork_prefix(it->first, prefix);
                oracle[id] = std::vector<kv::Token>(it->second.begin(), it->second.begin() + prefix);
            } else if (operation == 1) {
                std::vector<kv::Token> values(1 + generator() % 12, step);
                auto refs_before = pool.references();
                auto table_before = pool.sequence(it->first).blocks;
                try {
                    pool.append(it->first, values);
                    it->second.insert(it->second.end(), values.begin(), values.end());
                } catch (const std::bad_alloc&) {
                    require(pool.references() == refs_before, "random OOM references");
                    require(pool.sequence(it->first).blocks == table_before, "random OOM table");
                }
            } else {
                pool.release(it->first);
                oracle.erase(it);
            }
            pool.validate();
            for (const auto& item : oracle) require(pool.read(item.first) == item.second, "differential contents");
        }
        for (const auto& item : oracle) pool.release(item.first);
        require(pool.free_blocks() == 32, "random reclamation");
        pool.validate();
    }
    std::cout << "PASS: sharing, COW, release, boundaries, OOM atomicity, 2500-step differential test\n";
}
