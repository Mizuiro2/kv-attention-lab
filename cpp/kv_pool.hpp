#pragma once
#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <vector>

namespace kv {
using Token = std::int64_t;
using SequenceId = std::size_t;
struct Sequence { std::vector<std::size_t> blocks; std::size_t length = 0; };
struct Counters {
    std::size_t allocations = 0, frees = 0, copy_on_write = 0;
    std::size_t copied_token_slots = 0, high_water_blocks = 0;
};

// A control-plane allocator with token markers, not GPU K/V tensors or an attention kernel.
// Sequence IDs own references. The pool is deliberately single-threaded.
class BlockPool {
public:
    BlockPool(std::size_t blocks, std::size_t block_size)
        : block_size_(block_size), references_(blocks, 0), payload_(blocks), free_() {
        if (!blocks || !block_size) throw std::invalid_argument("positive pool dimensions required");
        free_.reserve(blocks);
        for (std::size_t i = blocks; i > 0; --i) {
            free_.push_back(i - 1);
            payload_[i - 1].resize(block_size, -1);
        }
    }

    SequenceId create() {
        auto id = next_id_;
        sequences_.emplace(id, Sequence{});
        ++next_id_;
        return id;
    }

    SequenceId fork_prefix(SequenceId source, std::size_t length) {
        const auto& parent = sequences_.at(source);
        if (length > parent.length) throw std::invalid_argument("prefix exceeds source");
        Sequence child;
        child.length = length;
        child.blocks.assign(parent.blocks.begin(), parent.blocks.begin() + ceil_blocks(length));
        auto id = next_id_;
        auto inserted = sequences_.emplace(id, std::move(child));
        // The map insertion is the last operation that can allocate.
        for (auto block : inserted.first->second.blocks) ++references_[block];
        ++next_id_;
        return id;
    }

    void append(SequenceId id, const std::vector<Token>& tokens) {
        auto& seq = sequences_.at(id);
        if (tokens.empty()) return;
        if (tokens.size() > std::numeric_limits<std::size_t>::max() - seq.length)
            throw std::length_error("sequence length overflow");
        const auto new_length = seq.length + tokens.size();
        const bool needs_cow = seq.length % block_size_ && references_[seq.blocks.back()] > 1;
        const auto extra = ceil_blocks(new_length) - seq.blocks.size() + (needs_cow ? 1 : 0);
        // Check capacity and reserve metadata before changing ANY references or payload.
        if (extra > free_.size()) throw std::bad_alloc();
        seq.blocks.reserve(ceil_blocks(new_length));
        if (needs_cow) {
            auto previous = seq.blocks.back();
            auto replacement = allocate();
            std::copy_n(payload_[previous].begin(), seq.length % block_size_, payload_[replacement].begin());
            --references_[previous]; // previous is shared, so it cannot become free here.
            seq.blocks.back() = replacement;
            ++counters_.copy_on_write;
            counters_.copied_token_slots += seq.length % block_size_;
        }
        for (auto token : tokens) {
            const auto offset = seq.length % block_size_;
            if (!offset) seq.blocks.push_back(allocate());
            payload_[seq.blocks.back()][offset] = token;
            ++seq.length;
        }
    }

    void release(SequenceId id) {
        const auto& seq = sequences_.at(id);
        for (auto block : seq.blocks) {
            if (--references_[block] == 0) {
                free_.push_back(block);
                ++counters_.frees;
            }
        }
        sequences_.erase(id);
    }

    std::vector<Token> read(SequenceId id) const {
        const auto& seq = sequences_.at(id);
        std::vector<Token> result;
        result.reserve(seq.length);
        for (std::size_t i = 0; i < seq.length; ++i)
            result.push_back(payload_[seq.blocks[i / block_size_]][i % block_size_]);
        return result;
    }

    // Count live physical slots once even when multiple sequences share a block.
    std::size_t used_token_slots() const {
        std::vector<std::size_t> used(references_.size(), 0);
        for (const auto& item : sequences_) {
            const auto& seq = item.second;
            for (std::size_t i = 0; i < seq.blocks.size(); ++i)
                used[seq.blocks[i]] = std::max(used[seq.blocks[i]], std::min(block_size_, seq.length - i * block_size_));
        }
        std::size_t total = 0;
        for (auto count : used) total += count;
        return total;
    }

    void validate() const {
        std::vector<std::size_t> expected(references_.size(), 0);
        for (const auto& item : sequences_) {
            const auto& seq = item.second;
            if (seq.blocks.size() != ceil_blocks(seq.length)) throw std::logic_error("block table length");
            for (auto block : seq.blocks) {
                if (block >= expected.size()) throw std::logic_error("block out of bounds");
                ++expected[block];
            }
        }
        if (expected != references_) throw std::logic_error("reference mismatch");
        std::vector<bool> seen(references_.size(), false);
        for (auto block : free_) {
            if (block >= seen.size() || seen[block] || references_[block]) throw std::logic_error("invalid free list");
            seen[block] = true;
        }
        for (std::size_t i = 0; i < references_.size(); ++i)
            if (seen[i] != (references_[i] == 0)) throw std::logic_error("lost block");
    }

    const Sequence& sequence(SequenceId id) const { return sequences_.at(id); }
    const std::vector<std::size_t>& references() const { return references_; }
    const Counters& counters() const { return counters_; }
    std::size_t free_blocks() const { return free_.size(); }
    std::size_t allocated_blocks() const { return references_.size() - free_.size(); }
    std::size_t allocated_token_capacity() const { return allocated_blocks() * block_size_; }
    std::size_t block_size() const { return block_size_; }
private:
    std::size_t ceil_blocks(std::size_t length) const { return length / block_size_ + (length % block_size_ != 0); }
    std::size_t allocate() {
        auto block = free_.back();
        free_.pop_back();
        references_[block] = 1;
        ++counters_.allocations;
        counters_.high_water_blocks = std::max(counters_.high_water_blocks, allocated_blocks());
        return block;
    }
    std::size_t block_size_, next_id_ = 0;
    std::vector<std::size_t> references_;
    std::vector<std::vector<Token>> payload_;
    std::vector<std::size_t> free_;
    std::map<SequenceId, Sequence> sequences_;
    Counters counters_;
};
} // namespace kv
