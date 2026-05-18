/*
 * bindings.cpp
 *
 * pybind11 bindings that expose chess_core and mcts_core to Python.
 *
 * Exposed Python names:
 *   chess_ext.Board         — full chess board
 *   chess_ext.MCTSNode      — single tree node
 *   chess_ext.MCTSTree      — tree manager
 *   chess_ext.init_tables() — explicit table initialisation (optional, auto-called)
 *   chess_ext.WHITE, BLACK, PAWN, ... — constants
 *   chess_ext.move_from, move_to, move_flags, make_move — move helpers
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "chess_core.h"
#include "mcts_core.h"

namespace py = pybind11;
using namespace chess;
using namespace mcts;

// ── Helper: Board → numpy float32 array (21, 8, 8) ───────────────────────

py::array_t<float> board_to_numpy(const Board &b) {
    auto result = py::array_t<float>({21, 8, 8});
    py::buffer_info buf = result.request();
    b.to_tensor(static_cast<float *>(buf.ptr));
    return result;
}

// ── Helper: numpy priors → std::vector<float> ────────────────────────────

std::vector<float> numpy_to_vector(py::array_t<float> arr) {
    py::buffer_info buf = arr.request();
    const float *ptr = static_cast<const float *>(buf.ptr);
    return std::vector<float>(ptr, ptr + buf.size);
}

// ============================================================
PYBIND11_MODULE(chess_ext, m) {
    m.doc() = "C++ chess engine + MCTS extension for Chess AI";

    // ── Global init ────────────────────────────────────────
    m.def("init_tables", &init_tables,
          "Initialise magic bitboard and Zobrist tables (called automatically).");

    // ── Constants ─────────────────────────────────────────
    m.attr("WHITE")  = WHITE;
    m.attr("BLACK")  = BLACK;
    m.attr("PAWN")   = PAWN;
    m.attr("KNIGHT") = KNIGHT;
    m.attr("BISHOP") = BISHOP;
    m.attr("ROOK")   = ROOK;
    m.attr("QUEEN")  = QUEEN;
    m.attr("KING")   = KING;
    m.attr("NO_SQ")  = NO_SQ;

    m.attr("CASTLE_WK") = CASTLE_WK;
    m.attr("CASTLE_WQ") = CASTLE_WQ;
    m.attr("CASTLE_BK") = CASTLE_BK;
    m.attr("CASTLE_BQ") = CASTLE_BQ;

    m.attr("PHASE_OPENING") = PHASE_OPENING;
    m.attr("PHASE_MID")     = PHASE_MID;
    m.attr("PHASE_ENDGAME") = PHASE_ENDGAME;

    m.attr("MT_QUIET")           = MT_QUIET;
    m.attr("MT_DOUBLE_PAWN")     = MT_DOUBLE_PAWN;
    m.attr("MT_CASTLE_K")        = MT_CASTLE_K;
    m.attr("MT_CASTLE_Q")        = MT_CASTLE_Q;
    m.attr("MT_CAPTURE")         = MT_CAPTURE;
    m.attr("MT_EP_CAPTURE")      = MT_EP_CAPTURE;
    m.attr("MT_PROMO_QUIET_Q")   = MT_PROMO_QUIET_Q;
    m.attr("MT_PROMO_CAPTURE_Q") = MT_PROMO_CAPTURE_Q;

    // ── Move helpers ───────────────────────────────────────
    m.def("move_from",  &move_from,  "Source square of a move (0-63).");
    m.def("move_to",    &move_to,    "Destination square of a move (0-63).");
    m.def("move_flags", &move_flags, "Move type flags (0-15).");
    m.def("make_move",  &make_move,  "Encode (from, to, flags) into a Move integer.");
    m.def("move_is_capture", &move_is_capture);
    m.def("move_is_promo",   &move_is_promo);
    m.def("promo_piece",     &promo_piece,
          "Promotion piece type (KNIGHT/BISHOP/ROOK/QUEEN).");

    // ── Board ──────────────────────────────────────────────
    py::class_<Board>(m, "Board")
        .def(py::init([]() {
            init_tables();
            return Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        }))
        .def_static("from_fen", [](const std::string &fen) {
            init_tables();
            return Board::from_fen(fen);
        }, py::arg("fen"), "Construct a Board from a FEN string.")
        .def("to_fen",           &Board::to_fen)
        .def("legal_moves",      &Board::legal_moves,
             "Return list of legal move integers.")
        .def("apply_move", [](const Board &b, Move m) { return b.apply_move(m); },
             py::arg("move"), "Return new Board after applying move.")
        .def("is_attacked",      &Board::is_attacked,
             py::arg("sq"), py::arg("attacker"))
        .def("in_check",         &Board::in_check)
        .def("is_checkmate",     &Board::is_checkmate)
        .def("is_stalemate",     &Board::is_stalemate)
        .def("is_fifty_move",    &Board::is_fifty_move)
        .def("is_threefold",     &Board::is_threefold)
        .def("is_insufficient_material", &Board::is_insufficient_material)
        .def("is_draw",          &Board::is_draw)
        .def("piece_count",      &Board::piece_count)
        .def("get_phase",        &Board::get_phase,
             "Returns PHASE_OPENING / PHASE_MID / PHASE_ENDGAME.")
        .def("piece_at",         &Board::piece_at,
             py::arg("sq"),
             "Piece index at square (color*6+type), or -1 if empty.")
        .def("to_tensor",        &board_to_numpy,
             "Return 21×8×8 float32 numpy array (neural net input planes).")
        .def("perft",            &Board::perft,
             py::arg("depth"), "Count leaf nodes at depth (move-gen verification).")
        .def("null_move",        &Board::null_move)
        .def_readonly("side_to_move",    &Board::side_to_move)
        .def_readonly("ep_square",       &Board::ep_square)
        .def_readonly("castling",        &Board::castling)
        .def_readonly("halfmove_clock",  &Board::halfmove_clock)
        .def_readonly("fullmove_number", &Board::fullmove_number)
        .def_readonly("hash",            &Board::hash)
        .def("__repr__", [](const Board &b) {
            return "<Board fen='" + b.to_fen() + "'>";
        });

    // ── MCTSNode (non-copyable — owns children via unique_ptr) ────────────
    py::class_<MCTSNode, std::unique_ptr<MCTSNode, py::nodelete>>(m, "MCTSNode")
        .def_readonly("N",             &MCTSNode::N)
        .def_readonly("W",             &MCTSNode::W)
        .def_readonly("Q",             &MCTSNode::Q)
        .def_readonly("P",             &MCTSNode::P)
        .def_readonly("move",          &MCTSNode::move)
        .def_readonly("terminal",      &MCTSNode::terminal)
        .def_readonly("terminal_value",&MCTSNode::terminal_value)
        .def("is_expanded",            &MCTSNode::is_expanded)
        .def("is_leaf",                &MCTSNode::is_leaf)
        .def("best_child",             &MCTSNode::best_child,
             py::arg("c_puct"), py::return_value_policy::reference)
        .def("sample_move",            &MCTSNode::sample_move,
             py::arg("temperature") = 1.0f)
        .def("get_policy_target",      &MCTSNode::get_policy_target,
             "Return list of (move, visit_fraction) pairs.")
        .def("get_value",              &MCTSNode::get_value)
        .def("get_children_count",     [](const MCTSNode &n) {
            return (int)n.children.size();
        })
        .def("get_child",              [](MCTSNode &n, Move m) -> MCTSNode * {
            auto it = n.children.find(m);
            if (it == n.children.end()) return nullptr;
            return it->second.get();
        }, py::arg("move"), py::return_value_policy::reference)
        .def("get_children",           [](MCTSNode &n) {
            std::vector<std::pair<Move, MCTSNode *>> result;
            for (auto &[mv, child] : n.children)
                result.push_back({mv, child.get()});
            return result;
        }, py::return_value_policy::reference)
        .def("__repr__", [](const MCTSNode &n) {
            return "<MCTSNode N=" + std::to_string(n.N)
                   + " Q=" + std::to_string(n.Q)
                   + " P=" + std::to_string(n.P) + ">";
        });

    // ── SelectionResult ────────────────────────────────────
    py::class_<SelectionResult>(m, "SelectionResult")
        .def_readonly("leaf",  &SelectionResult::leaf,
                      py::return_value_policy::reference)
        .def_readonly("board", &SelectionResult::board)
        .def("path_len",       [](const SelectionResult &r) {
            return (int)r.path.size();
        })
        .def("path",           [](const SelectionResult &r) {
            return r.path;
        }, py::return_value_policy::reference);

    // ── MCTSTree ───────────────────────────────────────────
    py::class_<MCTSTree>(m, "MCTSTree")
        .def(py::init<float>(), py::arg("c_puct") = 2.0f)
        .def("new_root",         [](MCTSTree &t, Move m, float p) {
            return t.new_root(m, p);
        }, py::arg("move") = 0, py::arg("prior") = 1.0f,
           py::return_value_policy::reference)
        .def("get_root",         &MCTSTree::get_root,
             py::return_value_policy::reference)
        .def("advance_root",     &MCTSTree::advance_root,
             py::arg("played_move"), py::return_value_policy::reference)
        .def("select_leaf",      &MCTSTree::select_leaf,
             py::arg("root_node"), py::arg("board"))
        .def("expand", [](MCTSTree &t, MCTSNode *leaf,
                           const std::vector<Move> &moves,
                           py::array_t<float> priors_arr) {
            auto priors = numpy_to_vector(priors_arr);
            t.expand(leaf, moves, priors);
        }, py::arg("leaf"), py::arg("legal_moves"), py::arg("priors"))
        .def("mark_terminal",    &MCTSTree::mark_terminal,
             py::arg("leaf"), py::arg("value"))
        .def("backup", [](MCTSTree &t,
                           const std::vector<MCTSNode *> &path,
                           float value, float td_lambda,
                           int move_number, int td_start, int td_end) {
            t.backup(path, value, td_lambda, move_number, td_start, td_end);
        }, py::arg("path"), py::arg("leaf_value"),
           py::arg("td_lambda")=0.8f, py::arg("move_number")=0,
           py::arg("td_start")=5, py::arg("td_end")=25)
        .def("add_dirichlet_noise", &MCTSTree::add_dirichlet_noise,
             py::arg("root_node"), py::arg("alpha")=0.3f, py::arg("epsilon")=0.25f)
        .def("tree_size",  [](MCTSTree &t) {
            return t.tree_size(t.get_root());
        })
        .def("average_depth", [](MCTSTree &t) {
            return t.average_depth(t.get_root(), 0);
        })
        .def_readwrite("c_puct", &MCTSTree::c_puct);
}
