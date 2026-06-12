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
#include <atomic>
#include <cassert>
#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
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
    // PERSPECTIVE: backup() accumulates each node's W/Q from the side-to-move
    // AT THAT NODE (the leaf gets +v_leaf, its parent -v_leaf, ...). The
    // parent selecting among children therefore needs the NEGATED child Q —
    // the child's side-to-move is the parent's opponent. Using +Q here made
    // the search maximise the opponent's value: a mating move was visited
    // once (terminal Q=-1 in the child's perspective) and then maximally
    // avoided (audit finding P3-5, proven in audit/scripts/test_puct_sign.py).
    // Each pending virtual loss additionally discounts the node by a full
    // loss so parallel leaf collection diversifies.
    float q_parent = -Q - (float)virtual_loss * 1.0f;
    return q_parent + u;
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

    // Compute visit count distribution with temperature. Work in log-space
    // and subtract the max before exponentiating: a direct N^(1/tau)
    // overflows float for tau in (0.01, ~0.05] (e.g. 128^50 ≈ 1e105 → inf,
    // degenerating the sampling; audit P3-3).
    std::vector<std::pair<chess::Move, float>> probs;
    probs.reserve(children.size());
    double max_log = -1e300;
    for (const auto &[mv, child_ptr] : children) {
        if (child_ptr->N > 0)
            max_log = std::max(max_log, std::log((double)child_ptr->N) / temperature);
    }
    float total = 0.0f;
    for (const auto &[mv, child_ptr] : children) {
        float p = 0.0f;
        if (child_ptr->N > 0)
            p = (float)std::exp(std::log((double)child_ptr->N) / temperature - max_log);
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
    // Runtime check (NOT assert — asserts are compiled out in release builds
    // and a too-short priors vector would read out of bounds; audit P3-2).
    if (legal_moves.size() != priors.size()) {
        throw std::invalid_argument(
            "MCTSTree::expand: legal_moves/priors size mismatch (" +
            std::to_string(legal_moves.size()) + " vs " +
            std::to_string(priors.size()) + ")");
    }
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

// ══════════════════════════════════════════════════════════════════════════
//  AlphaZero action encoding — C++ twin of movegen.py::move_to_action_index
// ══════════════════════════════════════════════════════════════════════════

int az_action_index(chess::Move m, int side_to_move) {
    int from  = (int)(m & 0x3F);
    int to    = (int)((m >> 6) & 0x3F);
    int flags = (int)((m >> 12) & 0xF);

    // Canonical (side-to-move) view: vertical flip for Black, matching
    // Board::to_tensor and the Python encoder (audit F1).
    if (side_to_move == 1) { from ^= 56; to ^= 56; }

    const int fr = from >> 3, ff = from & 7;
    const int tr = to >> 3,   tf = to & 7;
    const int dr = tr - fr,   df = tf - ff;

    constexpr int CH_PER_SQ = 73;
    constexpr int Q_CH = 56, K_CH = 8;

    // Underpromotions (N/B/R, quiet + capture): channels 64-72
    if (flags == 8 || flags == 9 || flags == 10 ||
        flags == 12 || flags == 13 || flags == 14) {
        int piece_idx = (flags == 8 || flags == 12) ? 0
                      : (flags == 9 || flags == 13) ? 1 : 2;  // N, B, R
        int dir_idx = (df == -1) ? 0 : (df == 0) ? 1 : 2;
        int channel = Q_CH + K_CH + piece_idx * 3 + dir_idx;
        return from * CH_PER_SQ + channel;
    }

    // Knight moves: channels 56-63
    static constexpr int KOFF[8][2] = {
        {2, 1}, {2, -1}, {-2, 1}, {-2, -1},
        {1, 2}, {1, -2}, {-1, 2}, {-1, -2},
    };
    for (int i = 0; i < 8; ++i)
        if (dr == KOFF[i][0] && df == KOFF[i][1])
            return from * CH_PER_SQ + Q_CH + i;

    // Sliding moves (queen promos included — same as a rank-8 queen move):
    // channels 0-55 (8 dirs × 7 distances)
    int dist = std::max(std::abs(dr), std::abs(df));
    if (dist == 0) return -1;
    if (dr != 0 && df != 0 && std::abs(dr) != std::abs(df)) return -1;
    const int ndr = dr ? dr / dist : 0;
    const int ndf = df ? df / dist : 0;
    static constexpr int QDIR[8][2] = {
        {0, 1}, {0, -1}, {1, 0}, {-1, 0},
        {1, 1}, {1, -1}, {-1, 1}, {-1, -1},
    };
    for (int d = 0; d < 8; ++d)
        if (ndr == QDIR[d][0] && ndf == QDIR[d][1])
            return from * CH_PER_SQ + d * 7 + (dist - 1);

    return -1;
}

// ══════════════════════════════════════════════════════════════════════════
//  RoundDriver — one batched simulation round entirely in C++
// ══════════════════════════════════════════════════════════════════════════

namespace {

// Simple fork-join over [0, n): trees are disjoint so per-index work is
// data-race-free. Exceptions are captured and rethrown on the caller thread.
template <typename Fn>
void parallel_for(int n, int n_threads, Fn &&fn) {
    if (n <= 0) return;
    int hw = (int)std::thread::hardware_concurrency();
    if (n_threads <= 0) n_threads = std::max(1, std::min(8, hw / 2));
    n_threads = std::min(n_threads, n);
    if (n_threads <= 1) {
        for (int i = 0; i < n; ++i) fn(i);
        return;
    }
    std::atomic<int> next{0};
    std::vector<std::thread> pool;
    std::mutex err_mu;
    std::string err;
    pool.reserve((size_t)n_threads);
    for (int t = 0; t < n_threads; ++t) {
        pool.emplace_back([&]() {
            try {
                int i;
                while ((i = next.fetch_add(1)) < n) fn(i);
            } catch (const std::exception &e) {
                std::lock_guard<std::mutex> lk(err_mu);
                if (err.empty()) err = e.what();
            } catch (...) {
                std::lock_guard<std::mutex> lk(err_mu);
                if (err.empty()) err = "unknown error in RoundDriver worker";
            }
        });
    }
    for (auto &th : pool) th.join();
    if (!err.empty()) throw std::runtime_error(err);
}

} // namespace

RoundDriver::RoundDriver(int num_actions) : num_actions_(num_actions) {}

int RoundDriver::select(const std::vector<MCTSTree *> &trees,
                        const std::vector<MCTSNode *> &roots,
                        const std::vector<const chess::Board *> &boards,
                        const std::vector<int> &fullmove_numbers,
                        float contempt,
                        float td_lambda, int td_start, int td_end,
                        float *out_boards, uint8_t *out_pending,
                        int n_threads) {
    const int n = (int)trees.size();
    if ((int)roots.size() != n || (int)boards.size() != n ||
        (int)fullmove_numbers.size() != n) {
        throw std::invalid_argument(
            "RoundDriver::select: trees/roots/boards/fullmoves length mismatch");
    }
    td_lambda_ = td_lambda;
    td_start_  = td_start;
    td_end_    = td_end;
    trees_     = trees;
    slots_.clear();
    slots_.resize((size_t)n);
    std::fill(out_pending, out_pending + n, (uint8_t)0);

    parallel_for(n, n_threads, [&](int k) {
        Slot &s = slots_[(size_t)k];
        SelectionResult res = trees[k]->select_leaf(roots[k], *boards[k]);
        s.path     = std::move(res.path);
        s.leaf     = res.leaf;
        s.board    = std::move(res.board);
        s.fullmove = fullmove_numbers[k];
        s.legal    = s.board.legal_moves();

        // Terminal / draw resolution — semantics of Python's
        // _resolve_non_nn + _finish_leaf, WITHOUT touching the
        // is_checkmate/is_stalemate predicates (their internal re-movegen
        // is the known native-crash path; legal.empty() + in_check() is
        // equivalent and safe):
        //   mate       (no legal, in check)  → value −1 for the mover
        //   stalemate  (no legal, no check)  → draw  (−contempt)
        //   fifty/threefold/insufficient     → draw  (−contempt),
        //     expanded with uniform priors when legal moves exist
        //     (preserves draw-with-legal-moves behaviour).
        bool  over  = false;
        float value = 0.0f;
        if (s.legal.empty()) {
            over  = true;
            value = s.board.in_check() ? -1.0f : -contempt;
        } else if (s.board.is_draw()) {
            over  = true;
            value = -contempt;
        }

        if (over) {
            if (s.leaf->terminal || s.legal.empty()) {
                trees[k]->mark_terminal(s.leaf, value);
            } else {
                std::vector<float> uniform(
                    s.legal.size(), 1.0f / (float)s.legal.size());
                trees[k]->expand(s.leaf, s.legal, uniform);
            }
            trees[k]->backup(s.path, value, td_lambda_, s.fullmove,
                             td_start_, td_end_);
            s.pending = false;
            return;
        }

        s.pending = true;
        out_pending[k] = 1;
        s.board.to_tensor(out_boards + (size_t)k * 21 * 64);
    });

    pending_idx_.clear();
    for (int k = 0; k < n; ++k)
        if (slots_[(size_t)k].pending) pending_idx_.push_back(k);
    return (int)pending_idx_.size();
}

void RoundDriver::finish(const float *policy, const float *values,
                         int n_threads) {
    const int p = (int)pending_idx_.size();

    parallel_for(p, n_threads, [&](int r) {
        const int k  = pending_idx_[(size_t)r];
        Slot     &s  = slots_[(size_t)k];
        MCTSTree *t  = trees_[(size_t)k];
        const float *row = policy + (size_t)r * (size_t)num_actions_;
        const float  val = values[r];

        // Priors — same numerics as policy_array_to_legal_priors():
        // gather (missing index → −1e9), max-subtract, softmax,
        // degenerate-sum → uniform.
        const size_t L = s.legal.size();
        std::vector<float> logits(L), priors(L);
        const int stm = s.board.side_to_move;
        float mx = -std::numeric_limits<float>::infinity();
        for (size_t i = 0; i < L; ++i) {
            int idx = az_action_index(s.legal[i], stm);
            float lg = (idx >= 0 && idx < num_actions_) ? row[idx] : -1e9f;
            logits[i] = lg;
            if (lg > mx) mx = lg;
        }
        float sum = 0.0f;
        for (size_t i = 0; i < L; ++i) {
            priors[i] = std::exp(logits[i] - mx);
            sum += priors[i];
        }
        if (sum <= 0.0f || !std::isfinite(sum)) {
            std::fill(priors.begin(), priors.end(), 1.0f / (float)L);
        } else {
            for (auto &q : priors) q /= sum;
        }

        if (s.leaf->terminal || s.legal.empty()) {
            t->mark_terminal(s.leaf, val);
        } else {
            t->expand(s.leaf, s.legal, priors);
        }
        t->backup(s.path, val, td_lambda_, s.fullmove, td_start_, td_end_);
    });

    pending_idx_.clear();
}

} // namespace mcts
