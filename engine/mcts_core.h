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

} // namespace mcts
