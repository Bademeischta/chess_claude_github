/*
 * mcts_core.h — MCTS node & tree public interface.
 */
#pragma once

#include "chess_core.h"

#include <memory>
#include <unordered_map>
#include <vector>

namespace mcts {

// ── MCTSNode ──────────────────────────────────────────────────────────────

struct MCTSNode {
    // Lifecycle canary (audit S1): constructed nodes carry ALIVE; the
    // destructor scribbles DEAD before releasing memory. Every API entry
    // point that walks node pointers validates the canary — a dangling
    // pointer (freed via advance_root/new_root but still referenced) is
    // then reported as a precise C++ exception at the touching site
    // instead of silently corrupting the heap and crashing minutes later
    // in an unrelated allocation (the 0xC0000005 crash class). Costs one
    // predictable compare next to reads the code does anyway.
    static constexpr uint32_t CANARY_ALIVE = 0xA11CE5EEu;
    static constexpr uint32_t CANARY_DEAD  = 0xDEADDEADu;
    uint32_t canary = CANARY_ALIVE;

    int   N;             // Visit count
    float W;             // Accumulated value (sum)
    float Q;             // Mean value = W / N
    float P;             // Prior probability from policy net
    int   virtual_loss;  // Threads deduct this during selection

    chess::Move  move;   // Move that led to this node (0 for root)
    MCTSNode    *parent; // Non-owning pointer to parent

    bool  terminal;
    float terminal_value;

    // Owning children: move → child node
    std::unordered_map<chess::Move, std::unique_ptr<MCTSNode>> children;

    explicit MCTSNode(chess::Move move = 0,
                      MCTSNode *parent = nullptr,
                      float prior = 1.0f);
    ~MCTSNode() { canary = CANARY_DEAD; }

    bool alive() const { return canary == CANARY_ALIVE; }

    // unique_ptr children make copy non-trivial; delete to prevent accidental copies.
    MCTSNode(const MCTSNode &) = delete;
    MCTSNode &operator=(const MCTSNode &) = delete;
    MCTSNode(MCTSNode &&) = default;
    MCTSNode &operator=(MCTSNode &&) = default;

    float ucb(float c_puct, int parent_n) const;
    void  add_virtual_loss();
    void  undo_virtual_loss();
    bool  is_expanded() const;
    bool  is_leaf() const;

    MCTSNode *best_child(float c_puct) const;
    chess::Move sample_move(float temperature) const;

    std::vector<std::pair<chess::Move, float>> get_policy_target() const;
    float get_value() const;
};

// ── SelectionResult ───────────────────────────────────────────────────────

struct SelectionResult {
    MCTSNode           *leaf;
    chess::Board        board;
    std::vector<MCTSNode *> path; // root → leaf (inclusive)
};

// ── MCTSTree ─────────────────────────────────────────────────────────────

class MCTSTree {
public:
    explicit MCTSTree(float c_puct = 2.0f);

    MCTSNode *new_root(chess::Move move = 0, float prior = 1.0f);
    MCTSNode *get_root() const;
    MCTSNode *advance_root(chess::Move played_move);

    // Selection phase: traverse tree to a leaf, apply virtual losses
    SelectionResult select_leaf(MCTSNode *root_node,
                                 const chess::Board &root_board);

    // Expansion: add children from network policy to a leaf node
    void expand(MCTSNode *leaf,
                const std::vector<chess::Move> &legal_moves,
                const std::vector<float> &priors);

    void mark_terminal(MCTSNode *leaf, float value);

    // Backup: propagate value up the path, undo virtual losses
    void backup(const std::vector<MCTSNode *> &path,
                float leaf_value,
                float td_lambda  = 0.8f,
                int   move_number = 0,
                int   td_start    = 5,
                int   td_end      = 25);

    // Add Dirichlet noise to root priors (called once per search)
    void add_dirichlet_noise(MCTSNode *root_node,
                              float alpha   = 0.3f,
                              float epsilon = 0.25f);

    // Diagnostics
    int   tree_size(MCTSNode *node) const;
    float average_depth(MCTSNode *node, int depth = 0) const;

    float c_puct;

private:
    std::unique_ptr<MCTSNode> root;
};

// ── AlphaZero action encoding (canonical, side-to-move view) ─────────────
// C++ twin of engine/movegen.py::move_to_action_index — MUST stay bit-equal
// (pinned by tests/test_round_driver.py parity test). Black squares are
// vertically flipped (sq ^ 56) to match the canonical board encoding.
int az_action_index(chess::Move m, int side_to_move);

// ── RoundDriver — one batched simulation round entirely in C++ ──────────
//
// Replaces the per-tree Python loop in ParallelMCTS._batched_sims:
//   select():  for every tree, run selection (virtual loss applied),
//              resolve terminal/draw leaves inline (mark/expand + backup),
//              and encode pending leaf boards into a caller-provided
//              (N, 21, 8, 8) float32 buffer. Returns a pending mask.
//   finish():  consume the network's (P, num_actions) policy logits and
//              (P,) values for the pending slots in mask order — compute
//              legal-move priors (canonical action gather + softmax),
//              expand and back up.
// Both phases release the GIL and fan out over a thread pool; trees are
// disjoint, so per-tree work is data-race-free by construction.
class RoundDriver {
public:
    explicit RoundDriver(int num_actions);

    // Returns number of pending (NN-needed) slots. out_boards must hold
    // trees.size() * 21 * 64 floats; out_pending trees.size() bytes.
    int select(const std::vector<MCTSTree *> &trees,
               const std::vector<MCTSNode *> &roots,
               const std::vector<const chess::Board *> &boards,
               const std::vector<int> &fullmove_numbers,
               float contempt,
               float td_lambda, int td_start, int td_end,
               float *out_boards, uint8_t *out_pending,
               int n_threads);

    // policy: (P, num_actions) float32, values: (P,) float32 — rows in the
    // order of the STILL-pending slots after any resolve_pending() calls.
    void finish(const float *policy, const float *values, int n_threads);

    // FEN of a pending slot's leaf board — for Python-side tablebase probing
    // (Syzygy probing lives in python-chess). Empty string if out of range.
    std::string slot_fen(int slot) const;

    // Resolve a pending slot with an externally-supplied value (e.g. a
    // tablebase WDL in side-to-move perspective): expand with uniform priors
    // (or mark terminal) + backup, clear its pending flag, and drop it from
    // the finish() batch. Lets Syzygy endgame leaves bypass the NN without
    // disabling the whole driver (the previous design did the latter).
    void resolve_pending(int slot, float value);

private:
    struct Slot {
        std::vector<MCTSNode *> path;
        MCTSNode    *leaf = nullptr;
        chess::Board board;
        chess::MoveList legal;     // cached from select (reused in finish)
        int  fullmove = 0;
        bool pending  = false;
    };
    std::vector<Slot> slots_;
    std::vector<int>  pending_idx_;   // slot index per policy row
    std::vector<MCTSTree *> trees_;   // borrowed for the finish phase
    int   num_actions_;
    float td_lambda_ = 0.8f;
    int   td_start_  = 5;
    int   td_end_    = 25;
};

} // namespace mcts
