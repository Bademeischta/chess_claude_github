/*
 * chess_core.h — public interface for the C++ chess engine.
 */
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace chess {

// ── Basic types ───────────────────────────────────────────────────────────

using Bitboard = uint64_t;
using Move     = uint32_t;
using MoveList = std::vector<Move>;

// ── Colors ────────────────────────────────────────────────────────────────

static constexpr int WHITE = 0;
static constexpr int BLACK = 1;

// ── Piece types ───────────────────────────────────────────────────────────

static constexpr int PAWN   = 0;
static constexpr int KNIGHT = 1;
static constexpr int BISHOP = 2;
static constexpr int ROOK   = 3;
static constexpr int QUEEN  = 4;
static constexpr int KING   = 5;

// ── Named squares ─────────────────────────────────────────────────────────

static constexpr int A1=0, B1=1, C1=2, D1=3, E1=4, F1=5, G1=6, H1=7;
static constexpr int A8=56,B8=57,C8=58,D8=59,E8=60,F8=61,G8=62,H8=63;
static constexpr int NO_SQ = -1;

// ── Castling rights bits ─────────────────────────────────────────────────

static constexpr int CASTLE_WK = 1;
static constexpr int CASTLE_WQ = 2;
static constexpr int CASTLE_BK = 4;
static constexpr int CASTLE_BQ = 8;

// ── Move type flags (4 bits in move encoding) ────────────────────────────

static constexpr int MT_QUIET            = 0;
static constexpr int MT_DOUBLE_PAWN      = 1;
static constexpr int MT_CASTLE_K         = 2;
static constexpr int MT_CASTLE_Q         = 3;
static constexpr int MT_CAPTURE          = 4;
static constexpr int MT_EP_CAPTURE       = 5;
// Promotion quiet: N=8, B=9, R=10, Q=11
static constexpr int MT_PROMO_QUIET_N    = 8;
static constexpr int MT_PROMO_QUIET_B    = 9;
static constexpr int MT_PROMO_QUIET_R    = 10;
static constexpr int MT_PROMO_QUIET_Q    = 11;
// Promotion capture: N=12, B=13, R=14, Q=15
static constexpr int MT_PROMO_CAPTURE_N  = 12;
static constexpr int MT_PROMO_CAPTURE_B  = 13;
static constexpr int MT_PROMO_CAPTURE_R  = 14;
static constexpr int MT_PROMO_CAPTURE_Q  = 15;

// ── Game phases ───────────────────────────────────────────────────────────

static constexpr int PHASE_OPENING  = 0;
static constexpr int PHASE_MID      = 1;
static constexpr int PHASE_ENDGAME  = 2;

// ── Move helpers ──────────────────────────────────────────────────────────

Move  make_move(int from, int to, int flags);
int   move_from(Move m);
int   move_to(Move m);
int   move_flags(Move m);
bool  move_is_capture(Move m);
bool  move_is_promo(Move m);
int   promo_piece(Move m);

// ── Board ─────────────────────────────────────────────────────────────────

struct Board {
    Bitboard pieces[12];       // [color*6 + piece_type]
    Bitboard occupied;
    Bitboard occupied_by[2];   // [color]
    int  side_to_move;
    int  ep_square;            // NO_SQ if none
    int  castling;             // CASTLE_* bits
    int  halfmove_clock;
    int  fullmove_number;
    uint64_t hash;             // Zobrist hash (also serves as repetition key)
    std::vector<uint64_t> hash_history;

    Board();
    void clear();

    static Board from_fen(const std::string &fen);
    std::string   to_fen() const;

    // Attack / check queries
    bool is_attacked(int sq, int attacker) const;
    bool in_check() const;

    // Move generation
    MoveList legal_moves() const;
    void     generate_pseudo(MoveList &ml) const;

    // Apply a move, returning a new board (non-destructive)
    Board apply_move(Move m) const;
    Board null_move() const;

    // Game state
    bool is_checkmate() const;
    bool is_stalemate() const;
    bool is_fifty_move() const;
    bool is_threefold() const;
    bool is_insufficient_material() const;
    bool is_draw() const;

    // Board info
    int  piece_count() const;
    int  get_phase() const;     // PHASE_OPENING / PHASE_MID / PHASE_ENDGAME
    int  piece_at(int sq) const;

    // Neural-network encoding: writes 21*64 floats into `out`
    void to_tensor(float *out) const;

    // Perft (bulk-count leaf nodes for move-gen verification)
    long long perft(int depth) const;

private:
    void put_piece(int piece_idx, int sq);
    void remove_piece(int piece_idx, int sq);
};

// Global table initialisation (called automatically, safe to call repeatedly)
void init_tables();

} // namespace chess
