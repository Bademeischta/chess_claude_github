/*
 * mcts_core.cpp
 *
 * C++ MCTS node & tree implementation.
 *
 * Design contract with Python:
 *   1. Python calls MCTSTree::select_leaf(root, board) → (leaf_node, leaf_board, path)
 *   2. Python evaluates the leaf with the neural network → (policy_arr, value)
 *   3. Python calls MCTSTree::expand(leaf_node, policy_arr, legal_moves)
 *   4. Python calls MCTSTree::backup(path, value, td_lambda)
 *
 * The tree owns all MCTSNode objects via unique_ptr.
 * Virtual loss prevents multiple threads from selecting the same leaf
 * (used when 8 parallel games batch their leaf evaluations).
 */

#include "mcts_core.h"
#include "chess_core.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <memory>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace mcts {

// Single high-quality PRNG (thread-local: select_leaf/backup may run from
// multiple game-driver threads). Replaces std::rand() which is global,
// low-quality and not thread-safe.
static std::mt19937_64 &rng() {
    static thread_local std::mt19937_64 g(
        std::random_device{}() ^ 0x9E3779B97F4A7C15ULL);
    return g;
}

// ============================================================
//  MCTSNode
// ============================================================

MCTSNode::MCTSNode(chess::Move move, MCTSNode *parent, float prior)
    : N(0), W(0.0f), Q(0.0f), P(prior),
      virtual_loss(0), move(move), parent(parent),
      terminal(false), terminal_value(0.0f) {}

float MCTSNode::ucb(float c_puct, int parent_n) const {
    float u = c_puct * P * std::sqrt((float)parent_n) / (1.0f + (float)(N + virtual_loss));
    float q_adj = Q - (float)virtual_loss * 1.0f; // each virtual loss reduces Q by 1
    return q_adj + u;
}

void MCTSNode::add_virtual_loss() {
    ++virtual_loss;
}

void MCTSNode::undo_virtual_loss() {
    if (virtual_loss > 0) --virtual_loss;
}

bool MCTSNode::is_expanded() const {
    return !children.empty() || terminal;
}

bool MCTSNode::is_leaf() const {
    return !is_expanded();
}

MCTSNode *MCTSNode::best_child(float c_puct) const {
    // RUNTIME guard, not assert: assert() is compiled out in release builds
    // (-DNDEBUG) and the loop below then iterates over an empty map and
    // returns nullptr, which the caller dereferences -> access violation.
    if (children.empty()) return nullptr;
    MCTSNode *best = nullptr;
    float best_score = -1e30f;
    int total_n = N;
    for (const auto &[mv, child_ptr] : children) {
        float score = child_ptr->ucb(c_puct, total_n);
        if (score > best_score) {
            best_score = score;
            best = child_ptr.get();
        }
    }
    return best;
}

// Temperature-sampled move selection (used for self-play action choice)
chess::Move MCTSNode::sample_move(float temperature) const {
    // RUNTIME guard, not assert: see best_child() above for rationale.
    // Return the invalid sentinel move 0 so the Python caller can detect
    // the degenerate state and recover via a legal-move fallback.
    if (children.empty()) return chess::Move(0);
    if (temperature <= 0.01f) {
        // Greedy: pick most visited child
        chess::Move best_move = 0;
        int best_n = -1;
        for (const auto &[mv, child_ptr] : children) {
            if (child_ptr->N > best_n) {
                best_n = child_ptr->N;
                best_move = mv;
            }
        }
        return best_move;
    }

    // Compute visit count distribution with temperature
    std::vector<std::pair<chess::Move, float>> probs;
    probs.reserve(children.size());
    float total = 0.0f;
    for (const auto &[mv, child_ptr] : children) {
        float p = std::pow((float)child_ptr->N, 1.0f / temperature);
        probs.push_back({mv, p});
        total += p;
    }
    // Normalise and sample
    std::uniform_real_distribution<float> dist(0.0f, total);
    float r = dist(rng());
    float cumulative = 0.0f;
    for (auto &[mv, p] : probs) {
        cumulative += p;
        if (r <= cumulative) return mv;
    }
    return probs.back().first;
}

// Get visit-count distribution over legal moves as policy target
std::vector<std::pair<chess::Move, float>> MCTSNode::get_policy_target() const {
    std::vector<std::pair<chess::Move, float>> result;
    result.reserve(children.size());
    float total = 0.0f;
    for (const auto &[mv, child_ptr] : children)
        total += (float)child_ptr->N;
    if (total < 1e-6f) total = 1.0f;
    for (const auto &[mv, child_ptr] : children)
        result.push_back({mv, (float)child_ptr->N / total});
    return result;
}

float MCTSNode::get_value() const {
    return (N > 0) ? W / (float)N : 0.0f;
}

// ============================================================
//  MCTSTree
// ============================================================

MCTSTree::MCTSTree(float c_puct)
    : c_puct(c_puct) {}

// Create a fresh root node for a new search
MCTSNode *MCTSTree::new_root(chess::Move move, float prior) {
    root = std::make_unique<MCTSNode>(move, nullptr, prior);
    return root.get();
}

MCTSNode *MCTSTree::get_root() const {
    return root.get();
}

// Move the tree root to the child corresponding to `played_move`.
// This preserves the subtree so future searches benefit from prior work.
MCTSNode *MCTSTree::advance_root(chess::Move played_move) {
    auto it = root->children.find(played_move);
    if (it == root->children.end()) {
        // Move was not explored during search — create a fresh root
        root = std::make_unique<MCTSNode>(played_move, nullptr, 1.0f);
        return root.get();
    }
    // Transfer ownership of the child subtree to be the new root
    std::unique_ptr<MCTSNode> new_root_ptr = std::move(it->second);
    new_root_ptr->parent = nullptr;
    root = std::move(new_root_ptr);
    return root.get();
}

// ── Selection ─────────────────────────────────────────────────────────────

SelectionResult MCTSTree::select_leaf(MCTSNode *root_node,
                                       const chess::Board &root_board) {
    SelectionResult result;
    result.path.push_back(root_node);
    MCTSNode *node = root_node;
    chess::Board board = root_board;

    while (node->is_expanded() && !node->terminal) {
        MCTSNode *child = node->best_child(c_puct);
        child->add_virtual_loss();
        result.path.push_back(child);
        board = board.apply_move(child->move);
        node = child;
    }

    result.leaf = node;
    result.board = board;
    return result;
}

// ── Expansion ────────────────────────────────────────────────────────────

void MCTSTree::expand(MCTSNode *leaf,
                       const std::vector<chess::Move> &legal_moves,
                       const std::vector<float> &priors) {
    if (leaf->terminal) return;
    if (legal_moves.empty()) {
        // Game over — terminal node
        leaf->terminal = true;
        return;
    }
    assert(legal_moves.size() == priors.size());
    // Double-expansion guard: if this leaf was already expanded (e.g. by a
    // parallel selection that raced past virtual_loss, or a Python caller
    // that re-invoked expand on the same node), `children[mv] = unique_ptr`
    // below would overwrite the existing child and silently leak its entire
    // subtree. `try_emplace` is a no-op when the key exists, preserving the
    // first expansion and its visit counts. The freshly-built node is
    // destroyed via unique_ptr if the insertion fails.
    for (size_t i = 0; i < legal_moves.size(); ++i) {
        chess::Move mv = legal_moves[i];
        auto fresh = std::make_unique<MCTSNode>(mv, leaf, priors[i]);
        leaf->children.try_emplace(mv, std::move(fresh));
    }
}

// Mark a leaf as a terminal position with a known value
void MCTSTree::mark_terminal(MCTSNode *leaf, float value) {
    leaf->terminal = true;
    leaf->terminal_value = value;
}

// ── Backup ────────────────────────────────────────────────────────────────

void MCTSTree::backup(const std::vector<MCTSNode *> &path,
                       float leaf_value,
                       float td_lambda,
                       int move_number,
                       int td_start,
                       int td_end) {
    // Determine whether to use TD(λ) blending or pure MC return.
    // path[0] = root, path.back() = leaf.
    // `leaf_value` is from the perspective of the side to move at the leaf.

    int path_len = (int)path.size();

    // Pure MC: propagate leaf_value back, alternating sign
    // TD(λ): blend with bootstrapped values (future work: compute n-step returns)
    // For simplicity: use pure MC propagation.  TD(λ) is applied at the
    // training level (trainer.py blends game outcome with MCTS Q-values).

    float value = leaf_value;
    for (int i = path_len - 1; i >= 0; --i) {
        MCTSNode *node = path[i];
        node->undo_virtual_loss();
        node->W += value;
        ++node->N;
        node->Q = node->W / (float)node->N;
        value = -value; // Flip perspective
    }
}

// ── Dirichlet noise ──────────────────────────────────────────────────────

void MCTSTree::add_dirichlet_noise(MCTSNode *root_node,
                                    float alpha,
                                    float epsilon) {
    if (root_node->children.empty()) return;
    int n = (int)root_node->children.size();

    // Proper Dirichlet(alpha,...,alpha): draw X_i ~ Gamma(alpha,1), then
    // normalise. The previous code used -log(u)*alpha = Exponential·alpha =
    // Gamma(1)·alpha, which is NOT Gamma(alpha) — it produced near-uniform
    // noise instead of the sparse Dirichlet(0.3) AlphaZero relies on for
    // root exploration.
    std::gamma_distribution<double> gamma(alpha, 1.0);
    std::vector<float> noise(n);
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        double g = gamma(rng());
        noise[i] = (float)g;
        sum += g;
    }
    if (sum < 1e-9) return;
    for (float &x : noise) x = (float)(x / sum);

    // Mix into prior probabilities
    int idx = 0;
    for (auto &[mv, child_ptr] : root_node->children) {
        child_ptr->P = (1.0f - epsilon) * child_ptr->P + epsilon * noise[idx];
        ++idx;
    }
}

// ── Statistics ────────────────────────────────────────────────────────────

int MCTSTree::tree_size(MCTSNode *node) const {
    if (!node) return 0;
    int count = 1;
    for (const auto &[mv, child] : node->children)
        count += tree_size(child.get());
    return count;
}

float MCTSTree::average_depth(MCTSNode *node, int depth) const {
    if (node->children.empty()) return (float)depth;
    float total = 0.0f;
    for (const auto &[mv, child] : node->children)
        total += average_depth(child.get(), depth + 1);
    return total / (float)node->children.size();
}

} // namespace mcts
