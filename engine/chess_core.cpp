/*
 * chess_core.cpp
 *
 * Complete chess engine with:
 *   - 12-bitboard representation (one per piece-type / color)
 *   - Magic bitboards for rook/bishop slider attacks
 *   - Lookup tables for knight, king, pawn attacks
 *   - Zobrist hashing (castling, ep, side-to-move, pieces)
 *   - Full legal move generation: pins, en passant, castling, promotions
 *   - FEN parsing and serialisation
 *   - 21-plane float tensor encoding for the neural network
 *   - Perft for move-generator verification
 *
 * Move encoding (uint32_t):
 *   bits  0- 5 : from square (0-63, a1=0 ... h8=63)
 *   bits  6-11 : to square
 *   bits 12-15 : move type flags (see MoveType enum)
 *
 * Square convention: a1=0, b1=1, ..., h1=7, a2=8, ..., h8=63
 * Piece array index: piece_idx = color*6 + piece_type
 *                    WHITE=0, BLACK=1; PAWN=0,KNIGHT=1,BISHOP=2,ROOK=3,QUEEN=4,KING=5
 */

#include "chess_core.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>

namespace chess {

// ============================================================
//  Constants & types
// ============================================================

static constexpr Bitboard FILE_A = 0x0101010101010101ULL;
static constexpr Bitboard FILE_H = 0x8080808080808080ULL;
static constexpr Bitboard RANK_1 = 0x00000000000000FFULL;
static constexpr Bitboard RANK_2 = 0x000000000000FF00ULL;
static constexpr Bitboard RANK_7 = 0x00FF000000000000ULL;
static constexpr Bitboard RANK_8 = 0xFF00000000000000ULL;
static constexpr Bitboard FULL   = 0xFFFFFFFFFFFFFFFFULL;

// ============================================================
//  Bitboard utilities
// ============================================================

inline int popcount(Bitboard b) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcountll(b);
#elif defined(_MSC_VER)
    return (int)__popcnt64(b);
#else
    int c = 0;
    while (b) { b &= b - 1; ++c; }
    return c;
#endif
}

inline int lsb(Bitboard b) {
    assert(b);
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_ctzll(b);
#elif defined(_MSC_VER)
    unsigned long idx;
    _BitScanForward64(&idx, b);
    return (int)idx;
#else
    int n = 0;
    if (!(b & 0xFFFFFFFFULL)) { n += 32; b >>= 32; }
    if (!(b & 0x0000FFFFULL)) { n += 16; b >>= 16; }
    if (!(b & 0x000000FFULL)) { n +=  8; b >>=  8; }
    if (!(b & 0x0000000FULL)) { n +=  4; b >>=  4; }
    if (!(b & 0x00000003ULL)) { n +=  2; b >>=  2; }
    if (!(b & 0x00000001ULL)) { n +=  1; }
    return n;
#endif
}

inline Bitboard pop_lsb(Bitboard &b) {
    Bitboard bit = b & (~b + 1ULL);   // isolate LSB (avoids MSVC C4146)
    b &= b - 1;
    return bit;
}

inline Bitboard sq_bit(int sq) { return 1ULL << sq; }
inline int rank_of(int sq)     { return sq >> 3; }
inline int file_of(int sq)     { return sq & 7; }
inline int make_sq(int r, int f){ return r * 8 + f; }

// ============================================================
//  Pseudo-random number generator (Xorshift64) for Zobrist + magic gen
// ============================================================

static uint64_t prng_state = 0xDEADBEEFCAFEBABEULL;

static uint64_t prng64() {
    prng_state ^= prng_state >> 12;
    prng_state ^= prng_state << 25;
    prng_state ^= prng_state >> 27;
    return prng_state * 0x2545F4914F6CDD1DULL;
}

static uint64_t sparse64() {
    return prng64() & prng64() & prng64();
}

// ============================================================
//  Zobrist tables
// ============================================================

static uint64_t ZOB_PIECE[12][64];  // [piece_idx][sq]
static uint64_t ZOB_CASTLING[16];   // 4-bit castling rights
static uint64_t ZOB_EP[9];          // ep file 0-7, index 8 = no ep
static uint64_t ZOB_BLACK_TO_MOVE;

static bool zobrist_initialised = false;

static void init_zobrist() {
    if (zobrist_initialised) return;
    prng_state = 0x0F0E0D0C0B0A0908ULL;
    for (int p = 0; p < 12; ++p)
        for (int sq = 0; sq < 64; ++sq)
            ZOB_PIECE[p][sq] = prng64();
    for (int c = 0; c < 16; ++c)
        ZOB_CASTLING[c] = prng64();
    for (int f = 0; f < 9; ++f)
        ZOB_EP[f] = prng64();
    ZOB_BLACK_TO_MOVE = prng64();
    zobrist_initialised = true;
}

// ============================================================
//  Precomputed attack tables (knights, kings, pawns)
// ============================================================

static Bitboard KNIGHT_ATTACKS[64];
static Bitboard KING_ATTACKS[64];
static Bitboard PAWN_ATTACKS[2][64];  // [color][sq]

static void init_attack_tables() {
    for (int sq = 0; sq < 64; ++sq) {
        int r = rank_of(sq), f = file_of(sq);
        Bitboard b = 0;

        // Knight
        const int dr[] = {2,2,-2,-2,1,-1,1,-1};
        const int df[] = {1,-1,1,-1,2,2,-2,-2};
        for (int i = 0; i < 8; ++i) {
            int nr = r + dr[i], nf = f + df[i];
            if (nr>=0 && nr<8 && nf>=0 && nf<8)
                b |= sq_bit(make_sq(nr, nf));
        }
        KNIGHT_ATTACKS[sq] = b;

        // King
        b = 0;
        for (int dr2 = -1; dr2 <= 1; ++dr2)
            for (int df2 = -1; df2 <= 1; ++df2) {
                if (!dr2 && !df2) continue;
                int nr = r + dr2, nf = f + df2;
                if (nr>=0 && nr<8 && nf>=0 && nf<8)
                    b |= sq_bit(make_sq(nr, nf));
            }
        KING_ATTACKS[sq] = b;

        // Pawn attacks (white attacks up-left / up-right)
        b = 0;
        if (r < 7) {
            if (f > 0) b |= sq_bit(sq + 7);
            if (f < 7) b |= sq_bit(sq + 9);
        }
        PAWN_ATTACKS[WHITE][sq] = b;

        // Pawn attacks (black attacks down-left / down-right)
        b = 0;
        if (r > 0) {
            if (f > 0) b |= sq_bit(sq - 9);
            if (f < 7) b |= sq_bit(sq - 7);
        }
        PAWN_ATTACKS[BLACK][sq] = b;
    }
}

// ============================================================
//  Magic Bitboards
// ============================================================

struct MagicEntry {
    Bitboard mask;
    Bitboard magic;
    int shift;
    Bitboard attacks[4096]; // max occupancy variants = 2^12
};

static MagicEntry ROOK_MAGIC[64];
static MagicEntry BISHOP_MAGIC[64];
static bool magic_initialised = false;

// Compute sliding attacks on a ray (used during magic generation)
static Bitboard rook_attacks_slow(int sq, Bitboard occ) {
    Bitboard result = 0;
    int r = rank_of(sq), f = file_of(sq);
    // North
    for (int nr = r+1; nr < 8; ++nr) {
        result |= sq_bit(make_sq(nr, f));
        if (occ & sq_bit(make_sq(nr, f))) break;
    }
    // South
    for (int nr = r-1; nr >= 0; --nr) {
        result |= sq_bit(make_sq(nr, f));
        if (occ & sq_bit(make_sq(nr, f))) break;
    }
    // East
    for (int nf = f+1; nf < 8; ++nf) {
        result |= sq_bit(make_sq(r, nf));
        if (occ & sq_bit(make_sq(r, nf))) break;
    }
    // West
    for (int nf = f-1; nf >= 0; --nf) {
        result |= sq_bit(make_sq(r, nf));
        if (occ & sq_bit(make_sq(r, nf))) break;
    }
    return result;
}

static Bitboard bishop_attacks_slow(int sq, Bitboard occ) {
    Bitboard result = 0;
    int r = rank_of(sq), f = file_of(sq);
    // NE
    for (int d = 1; r+d < 8 && f+d < 8; ++d) {
        result |= sq_bit(make_sq(r+d, f+d));
        if (occ & sq_bit(make_sq(r+d, f+d))) break;
    }
    // NW
    for (int d = 1; r+d < 8 && f-d >= 0; ++d) {
        result |= sq_bit(make_sq(r+d, f-d));
        if (occ & sq_bit(make_sq(r+d, f-d))) break;
    }
    // SE
    for (int d = 1; r-d >= 0 && f+d < 8; ++d) {
        result |= sq_bit(make_sq(r-d, f+d));
        if (occ & sq_bit(make_sq(r-d, f+d))) break;
    }
    // SW
    for (int d = 1; r-d >= 0 && f-d >= 0; ++d) {
        result |= sq_bit(make_sq(r-d, f-d));
        if (occ & sq_bit(make_sq(r-d, f-d))) break;
    }
    return result;
}

// Occupancy mask (blockers only — edge squares excluded, they don't block)
static Bitboard rook_mask(int sq) {
    Bitboard mask = rook_attacks_slow(sq, 0);
    // Remove edge squares from mask (they always contribute attacks)
    if (rank_of(sq) != 0) mask &= ~RANK_1;
    if (rank_of(sq) != 7) mask &= ~RANK_8;
    if (file_of(sq) != 0) mask &= ~FILE_A;
    if (file_of(sq) != 7) mask &= ~FILE_H;
    return mask;
}

static Bitboard bishop_mask(int sq) {
    Bitboard mask = bishop_attacks_slow(sq, 0);
    mask &= ~RANK_1 & ~RANK_8 & ~FILE_A & ~FILE_H;
    return mask;
}

// Enumerate all subsets of a mask (Carry-Rippler technique)
static void enum_subsets(Bitboard mask, std::vector<Bitboard> &out) {
    out.clear();
    Bitboard sub = 0;
    do {
        out.push_back(sub);
        sub = (sub - mask) & mask;
    } while (sub);
}

static bool try_magic(MagicEntry &me, int sq, bool rook) {
    int bits = popcount(me.mask);
    me.shift = 64 - bits;
    int size = 1 << bits;
    // Clear attacks table
    std::fill(me.attacks, me.attacks + size, Bitboard(0));
    std::vector<Bitboard> occ_list;
    occ_list.reserve(size);
    enum_subsets(me.mask, occ_list);
    // Verify magic maps each occupancy to unique slot
    std::vector<bool> used(size, false);
    for (Bitboard occ : occ_list) {
        int idx = (int)((occ * me.magic) >> me.shift);
        Bitboard atk = rook ? rook_attacks_slow(sq, occ) : bishop_attacks_slow(sq, occ);
        if (!used[idx]) {
            me.attacks[idx] = atk;
            used[idx] = true;
        } else if (me.attacks[idx] != atk) {
            return false; // Collision with different attack set
        }
    }
    return true;
}

static void generate_magic(MagicEntry &me, int sq, bool rook) {
    me.mask = rook ? rook_mask(sq) : bishop_mask(sq);
    for (;;) {
        me.magic = sparse64();
        if (try_magic(me, sq, rook)) return;
    }
}

static void init_magic_tables() {
    if (magic_initialised) return;
    prng_state = 0xCAFEBABEDEAD1234ULL; // Different seed from Zobrist
    for (int sq = 0; sq < 64; ++sq) {
        generate_magic(ROOK_MAGIC[sq], sq, true);
        generate_magic(BISHOP_MAGIC[sq], sq, false);
    }
    magic_initialised = true;
}

// ── Fast magic lookup ──────────────────────────────────────

inline Bitboard rook_attacks(int sq, Bitboard occ) {
    const MagicEntry &me = ROOK_MAGIC[sq];
    return me.attacks[((occ & me.mask) * me.magic) >> me.shift];
}

inline Bitboard bishop_attacks(int sq, Bitboard occ) {
    const MagicEntry &me = BISHOP_MAGIC[sq];
    return me.attacks[((occ & me.mask) * me.magic) >> me.shift];
}

inline Bitboard queen_attacks(int sq, Bitboard occ) {
    return rook_attacks(sq, occ) | bishop_attacks(sq, occ);
}

// ============================================================
//  Global initialisation
// ============================================================

bool tables_initialised = false;

void init_tables() {
    if (tables_initialised) return;
    init_zobrist();
    init_attack_tables();
    init_magic_tables();
    tables_initialised = true;
}

// ============================================================
//  Move helpers
// ============================================================

Move make_move(int from, int to, int flags) {
    return (Move)((from) | (to << 6) | (flags << 12));
}

int move_from(Move m)  { return m & 0x3F; }
int move_to(Move m)    { return (m >> 6) & 0x3F; }
int move_flags(Move m) { return (m >> 12) & 0xF; }

bool move_is_capture(Move m) {
    int f = move_flags(m);
    return f == MT_CAPTURE || f == MT_EP_CAPTURE ||
           f == MT_PROMO_CAPTURE_N || f == MT_PROMO_CAPTURE_B ||
           f == MT_PROMO_CAPTURE_R || f == MT_PROMO_CAPTURE_Q;
}

bool move_is_promo(Move m) {
    int f = move_flags(m);
    return f >= MT_PROMO_QUIET_N;
}

int promo_piece(Move m) {
    // Returns KNIGHT/BISHOP/ROOK/QUEEN
    int f = move_flags(m);
    switch (f) {
        case MT_PROMO_QUIET_N: case MT_PROMO_CAPTURE_N: return KNIGHT;
        case MT_PROMO_QUIET_B: case MT_PROMO_CAPTURE_B: return BISHOP;
        case MT_PROMO_QUIET_R: case MT_PROMO_CAPTURE_R: return ROOK;
        default:                                          return QUEEN;
    }
}

// ============================================================
//  Board: construction and helpers
// ============================================================

Board::Board() {
    init_tables();
    clear();
}

void Board::clear() {
    std::fill(pieces, pieces + 12, Bitboard(0));
    occupied = 0;
    occupied_by[WHITE] = 0;
    occupied_by[BLACK] = 0;
    side_to_move = WHITE;
    ep_square = NO_SQ;
    castling = 0;
    halfmove_clock = 0;
    fullmove_number = 1;
    hash = 0;
    hash_history.clear();
}

// Not `inline`: bindings.cpp takes &Board::piece_at in another TU, so this
// must have a single external definition (ODR). /GL+/LTCG still inlines it.
int Board::piece_at(int sq) const {
    Bitboard bit = sq_bit(sq);
    for (int p = 0; p < 12; ++p)
        if (pieces[p] & bit) return p;
    return -1;
}

void Board::put_piece(int piece_idx, int sq) {
    Bitboard bit = sq_bit(sq);
    pieces[piece_idx] |= bit;
    int color = piece_idx / 6;
    occupied_by[color] |= bit;
    occupied |= bit;
    hash ^= ZOB_PIECE[piece_idx][sq];
}

void Board::remove_piece(int piece_idx, int sq) {
    Bitboard bit = sq_bit(sq);
    pieces[piece_idx] &= ~bit;
    int color = piece_idx / 6;
    occupied_by[color] &= ~bit;
    occupied &= ~bit;
    hash ^= ZOB_PIECE[piece_idx][sq];
}

// ============================================================
//  FEN parsing
// ============================================================

Board Board::from_fen(const std::string &fen) {
    Board b;
    b.clear();
    std::istringstream ss(fen);
    std::string token;

    // 1. Piece placement
    ss >> token;
    int sq = 56; // a8=56
    for (char c : token) {
        if (c == '/') {
            sq -= 16; // Go to start of next rank down
        } else if (c >= '1' && c <= '8') {
            sq += c - '0';
        } else {
            int color = (c >= 'a') ? BLACK : WHITE;
            int piece;
            char lc = (char)std::tolower((unsigned char)c);
            if      (lc == 'p') piece = PAWN;
            else if (lc == 'n') piece = KNIGHT;
            else if (lc == 'b') piece = BISHOP;
            else if (lc == 'r') piece = ROOK;
            else if (lc == 'q') piece = QUEEN;
            else if (lc == 'k') piece = KING;
            else { ++sq; continue; }
            b.put_piece(color * 6 + piece, sq);
            ++sq;
        }
    }

    // 2. Side to move
    ss >> token;
    b.side_to_move = (token == "b") ? BLACK : WHITE;
    if (b.side_to_move == BLACK) b.hash ^= ZOB_BLACK_TO_MOVE;

    // 3. Castling rights
    ss >> token;
    b.castling = 0;
    for (char c : token) {
        if      (c == 'K') b.castling |= CASTLE_WK;
        else if (c == 'Q') b.castling |= CASTLE_WQ;
        else if (c == 'k') b.castling |= CASTLE_BK;
        else if (c == 'q') b.castling |= CASTLE_BQ;
    }
    b.hash ^= ZOB_CASTLING[b.castling];

    // 4. En passant
    ss >> token;
    b.ep_square = NO_SQ;
    if (token.size() >= 2 && token != "-" &&
        token[0] >= 'a' && token[0] <= 'h' &&
        token[1] >= '1' && token[1] <= '8') {
        int ep_file = token[0] - 'a';
        int ep_rank = token[1] - '1';
        b.ep_square = make_sq(ep_rank, ep_file);
        b.hash ^= ZOB_EP[ep_file];
    } else {
        b.hash ^= ZOB_EP[8]; // no / malformed ep
    }

    // 5. Halfmove clock  6. Fullmove number — tolerate missing/garbage fields
    auto safe_stoi = [](const std::string &s, int def) -> int {
        try { return std::stoi(s); } catch (...) { return def; }
    };
    if (ss >> token) b.halfmove_clock  = safe_stoi(token, 0);
    if (ss >> token) b.fullmove_number = safe_stoi(token, 1);

    b.hash_history.push_back(b.hash);
    return b;
}

std::string Board::to_fen() const {
    std::string result;

    // Piece placement
    for (int r = 7; r >= 0; --r) {
        int empty = 0;
        for (int f = 0; f < 8; ++f) {
            int sq = make_sq(r, f);
            int p = piece_at(sq);
            if (p == -1) {
                ++empty;
            } else {
                if (empty) { result += char('0' + empty); empty = 0; }
                static const char piece_chars[] = "PNBRQKpnbrqk";
                result += piece_chars[p];
            }
        }
        if (empty) result += char('0' + empty);
        if (r > 0) result += '/';
    }

    result += ' ';
    result += (side_to_move == WHITE) ? 'w' : 'b';
    result += ' ';

    if (!castling) {
        result += '-';
    } else {
        if (castling & CASTLE_WK) result += 'K';
        if (castling & CASTLE_WQ) result += 'Q';
        if (castling & CASTLE_BK) result += 'k';
        if (castling & CASTLE_BQ) result += 'q';
    }

    result += ' ';
    if (ep_square == NO_SQ) {
        result += '-';
    } else {
        result += char('a' + file_of(ep_square));
        result += char('1' + rank_of(ep_square));
    }

    result += ' ';
    result += std::to_string(halfmove_clock);
    result += ' ';
    result += std::to_string(fullmove_number);

    return result;
}

// ============================================================
//  Attack detection
// ============================================================

// Is square `sq` attacked by `attacker` color?
bool Board::is_attacked(int sq, int attacker) const {
    int defender = 1 - attacker;
    Bitboard occ = occupied;

    // Pawn attacks
    if (PAWN_ATTACKS[defender][sq] & pieces[attacker * 6 + PAWN]) return true;
    // Knight
    if (KNIGHT_ATTACKS[sq] & pieces[attacker * 6 + KNIGHT]) return true;
    // King
    if (KING_ATTACKS[sq] & pieces[attacker * 6 + KING]) return true;
    // Bishops / Queens (diagonal)
    Bitboard diag = bishop_attacks(sq, occ);
    if (diag & (pieces[attacker * 6 + BISHOP] | pieces[attacker * 6 + QUEEN])) return true;
    // Rooks / Queens (orthogonal)
    Bitboard orth = rook_attacks(sq, occ);
    if (orth & (pieces[attacker * 6 + ROOK] | pieces[attacker * 6 + QUEEN])) return true;

    return false;
}

bool Board::in_check() const {
    Bitboard king = pieces[side_to_move * 6 + KING];
    if (!king) return false;          // malformed position — avoid lsb(0) UB
    return is_attacked(lsb(king), 1 - side_to_move);
}

// ============================================================
//  Pseudo-legal move generation helpers
// ============================================================

static void add_promos(MoveList &ml, int from, int to, bool capture) {
    int base = capture ? MT_PROMO_CAPTURE_N : MT_PROMO_QUIET_N;
    ml.push_back(make_move(from, to, base));
    ml.push_back(make_move(from, to, base + 1));
    ml.push_back(make_move(from, to, base + 2));
    ml.push_back(make_move(from, to, base + 3));
}

// ============================================================
//  Legal move generation
// ============================================================

MoveList Board::legal_moves() const {
    MoveList pseudo, legal;
    pseudo.reserve(64);
    legal.reserve(64);
    generate_pseudo(pseudo);

    for (Move m : pseudo) {
        Board next = apply_move(m);
        // After our move, OUR king must not be attacked by the OPPONENT
        int king_sq = lsb(next.pieces[side_to_move * 6 + KING]);
        if (!next.is_attacked(king_sq, 1 - side_to_move)) {
            legal.push_back(m);
        }
    }
    return legal;
}

void Board::generate_pseudo(MoveList &ml) const {
    int us   = side_to_move;
    int them = 1 - us;
    Bitboard our   = occupied_by[us];
    Bitboard their = occupied_by[them];

    // ── Pawns ──────────────────────────────────────────────────────────
    Bitboard pawns = pieces[us * 6 + PAWN];
    while (pawns) {
        int from = lsb(pawns); pawns &= pawns - 1;
        int r = rank_of(from), f = file_of(from);

        if (us == WHITE) {
            // Single push
            int to = from + 8;
            if (!(occupied & sq_bit(to))) {
                if (r == 6) add_promos(ml, from, to, false);
                else {
                    ml.push_back(make_move(from, to, MT_QUIET));
                    // Double push
                    if (r == 1 && !(occupied & sq_bit(to + 8)))
                        ml.push_back(make_move(from, to + 8, MT_DOUBLE_PAWN));
                }
            }
            // Captures
            Bitboard cap = PAWN_ATTACKS[WHITE][from] & their;
            while (cap) {
                int t = lsb(cap); cap &= cap - 1;
                if (rank_of(t) == 7) add_promos(ml, from, t, true);
                else ml.push_back(make_move(from, t, MT_CAPTURE));
            }
            // En passant
            if (ep_square != NO_SQ && (PAWN_ATTACKS[WHITE][from] & sq_bit(ep_square)))
                ml.push_back(make_move(from, ep_square, MT_EP_CAPTURE));
        } else {
            // Single push
            int to = from - 8;
            if (!(occupied & sq_bit(to))) {
                if (r == 1) add_promos(ml, from, to, false);
                else {
                    ml.push_back(make_move(from, to, MT_QUIET));
                    // Double push
                    if (r == 6 && !(occupied & sq_bit(to - 8)))
                        ml.push_back(make_move(from, to - 8, MT_DOUBLE_PAWN));
                }
            }
            // Captures
            Bitboard cap = PAWN_ATTACKS[BLACK][from] & their;
            while (cap) {
                int t = lsb(cap); cap &= cap - 1;
                if (rank_of(t) == 0) add_promos(ml, from, t, true);
                else ml.push_back(make_move(from, t, MT_CAPTURE));
            }
            // En passant
            if (ep_square != NO_SQ && (PAWN_ATTACKS[BLACK][from] & sq_bit(ep_square)))
                ml.push_back(make_move(from, ep_square, MT_EP_CAPTURE));
        }
    }

    // ── Knights ────────────────────────────────────────────────────────
    Bitboard knights = pieces[us * 6 + KNIGHT];
    while (knights) {
        int from = lsb(knights); knights &= knights - 1;
        Bitboard attacks = KNIGHT_ATTACKS[from] & ~our;
        while (attacks) {
            int to = lsb(attacks); attacks &= attacks - 1;
            int flags = (their & sq_bit(to)) ? MT_CAPTURE : MT_QUIET;
            ml.push_back(make_move(from, to, flags));
        }
    }

    // ── Bishops ────────────────────────────────────────────────────────
    Bitboard bishops = pieces[us * 6 + BISHOP];
    while (bishops) {
        int from = lsb(bishops); bishops &= bishops - 1;
        Bitboard attacks = bishop_attacks(from, occupied) & ~our;
        while (attacks) {
            int to = lsb(attacks); attacks &= attacks - 1;
            int flags = (their & sq_bit(to)) ? MT_CAPTURE : MT_QUIET;
            ml.push_back(make_move(from, to, flags));
        }
    }

    // ── Rooks ──────────────────────────────────────────────────────────
    Bitboard rooks = pieces[us * 6 + ROOK];
    while (rooks) {
        int from = lsb(rooks); rooks &= rooks - 1;
        Bitboard attacks = rook_attacks(from, occupied) & ~our;
        while (attacks) {
            int to = lsb(attacks); attacks &= attacks - 1;
            int flags = (their & sq_bit(to)) ? MT_CAPTURE : MT_QUIET;
            ml.push_back(make_move(from, to, flags));
        }
    }

    // ── Queens ─────────────────────────────────────────────────────────
    Bitboard queens = pieces[us * 6 + QUEEN];
    while (queens) {
        int from = lsb(queens); queens &= queens - 1;
        Bitboard attacks = queen_attacks(from, occupied) & ~our;
        while (attacks) {
            int to = lsb(attacks); attacks &= attacks - 1;
            int flags = (their & sq_bit(to)) ? MT_CAPTURE : MT_QUIET;
            ml.push_back(make_move(from, to, flags));
        }
    }

    // ── King ───────────────────────────────────────────────────────────
    {
        int from = lsb(pieces[us * 6 + KING]);
        Bitboard attacks = KING_ATTACKS[from] & ~our;
        while (attacks) {
            int to = lsb(attacks); attacks &= attacks - 1;
            int flags = (their & sq_bit(to)) ? MT_CAPTURE : MT_QUIET;
            ml.push_back(make_move(from, to, flags));
        }

        // Castling — legality (king not through check) verified in legal_moves()
        if (us == WHITE) {
            if ((castling & CASTLE_WK) &&
                !(occupied & (sq_bit(F1) | sq_bit(G1))) &&
                !is_attacked(E1, BLACK) &&
                !is_attacked(F1, BLACK) &&
                !is_attacked(G1, BLACK))
                ml.push_back(make_move(E1, G1, MT_CASTLE_K));

            if ((castling & CASTLE_WQ) &&
                !(occupied & (sq_bit(D1) | sq_bit(C1) | sq_bit(B1))) &&
                !is_attacked(E1, BLACK) &&
                !is_attacked(D1, BLACK) &&
                !is_attacked(C1, BLACK))
                ml.push_back(make_move(E1, C1, MT_CASTLE_Q));
        } else {
            if ((castling & CASTLE_BK) &&
                !(occupied & (sq_bit(F8) | sq_bit(G8))) &&
                !is_attacked(E8, WHITE) &&
                !is_attacked(F8, WHITE) &&
                !is_attacked(G8, WHITE))
                ml.push_back(make_move(E8, G8, MT_CASTLE_K));

            if ((castling & CASTLE_BQ) &&
                !(occupied & (sq_bit(D8) | sq_bit(C8) | sq_bit(B8))) &&
                !is_attacked(E8, WHITE) &&
                !is_attacked(D8, WHITE) &&
                !is_attacked(C8, WHITE))
                ml.push_back(make_move(E8, C8, MT_CASTLE_Q));
        }
    }
}

// ============================================================
//  Apply move → new Board
// ============================================================

Board Board::apply_move(Move m) const {
    Board next = *this;
    int from = move_from(m);
    int to   = move_to(m);
    int flags = move_flags(m);
    int us   = side_to_move;
    int them = 1 - us;

    // Clear ep hash contribution (will be re-added if new ep)
    next.hash ^= ZOB_EP[(ep_square == NO_SQ) ? 8 : file_of(ep_square)];
    next.ep_square = NO_SQ;

    // Remove castling hash, will re-add
    next.hash ^= ZOB_CASTLING[next.castling];

    // Identify moving piece
    int moving = -1;
    for (int p = us * 6; p < us * 6 + 6; ++p) {
        if (next.pieces[p] & sq_bit(from)) { moving = p; break; }
    }

    // Capture: remove victim. Covers normal captures AND promotion-captures
    // (flags 12-15); en passant is handled separately below. The promotion
    // branch later only PLACES the promoted piece, so the victim on `to`
    // must be removed here or it stays on the board (corrupting bitboards,
    // occupancy and the Zobrist hash).
    if (flags == MT_CAPTURE ||
        (flags >= MT_PROMO_CAPTURE_N && flags <= MT_PROMO_CAPTURE_Q)) {
        int victim = -1;
        for (int p = them * 6; p < them * 6 + 6; ++p) {
            if (next.pieces[p] & sq_bit(to)) { victim = p; break; }
        }
        if (victim >= 0) next.remove_piece(victim, to);
    }

    // En passant capture
    if (flags == MT_EP_CAPTURE) {
        int cap_sq = (us == WHITE) ? to - 8 : to + 8;
        next.remove_piece(them * 6 + PAWN, cap_sq);
    }

    // Move the piece
    next.remove_piece(moving, from);

    if (move_is_promo(m)) {
        // Victim (if any) was already removed in the capture branch above.
        next.put_piece(us * 6 + promo_piece(m), to);
    } else {
        next.put_piece(moving, to);
    }

    // Castling: also move the rook
    if (flags == MT_CASTLE_K) {
        if (us == WHITE) {
            next.remove_piece(WHITE * 6 + ROOK, H1);
            next.put_piece(WHITE * 6 + ROOK, F1);
        } else {
            next.remove_piece(BLACK * 6 + ROOK, H8);
            next.put_piece(BLACK * 6 + ROOK, F8);
        }
    }
    if (flags == MT_CASTLE_Q) {
        if (us == WHITE) {
            next.remove_piece(WHITE * 6 + ROOK, A1);
            next.put_piece(WHITE * 6 + ROOK, D1);
        } else {
            next.remove_piece(BLACK * 6 + ROOK, A8);
            next.put_piece(BLACK * 6 + ROOK, D8);
        }
    }

    // Double pawn push: only record the ep square if an enemy pawn can
    // actually capture en passant (standard / python-chess rule). Recording
    // a phantom ep otherwise would desync the Zobrist hash for functionally
    // identical positions (hurting threefold detection) and wrongly set the
    // FEN ep field / NN ep plane. PAWN_ATTACKS[us][ep] are exactly the squares
    // an enemy pawn must occupy to capture the pushed pawn.
    if (flags == MT_DOUBLE_PAWN) {
        int ep = (us == WHITE) ? from + 8 : from - 8;
        if (PAWN_ATTACKS[us][ep] & next.pieces[them * 6 + PAWN]) {
            next.ep_square = ep;
            next.hash ^= ZOB_EP[file_of(ep)];
        } else {
            next.hash ^= ZOB_EP[8]; // no capturable ep
        }
    } else {
        next.hash ^= ZOB_EP[8]; // no ep
    }

    // Update castling rights
    static const int CASTLING_MASK[64] = {
        ~CASTLE_WQ, 15, 15, 15, ~(CASTLE_WK|CASTLE_WQ), 15, 15, ~CASTLE_WK,
        15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,
        15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,
        15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,
        ~CASTLE_BQ, 15, 15, 15, ~(CASTLE_BK|CASTLE_BQ), 15, 15, ~CASTLE_BK,
    };
    next.castling &= CASTLING_MASK[from] & CASTLING_MASK[to];
    next.hash ^= ZOB_CASTLING[next.castling];

    // Halfmove clock: reset on capture or pawn move
    bool irreversible = (flags == MT_CAPTURE || flags == MT_EP_CAPTURE ||
                         (moving % 6) == PAWN || move_is_promo(m));
    if (irreversible)
        next.halfmove_clock = 0;
    else
        ++next.halfmove_clock;

    // Fullmove number
    if (us == BLACK) ++next.fullmove_number;

    // Switch side
    next.side_to_move = them;
    next.hash ^= ZOB_BLACK_TO_MOVE;

    // A repetition can never span an irreversible move (capture / pawn move /
    // promotion), so the repetition window is bounded by the halfmove clock.
    // Clearing here keeps is_threefold correct AND bounds hash_history to
    // ≤ halfmove_clock entries, so the per-node `Board next = *this` copy no
    // longer scales with full game length (big perft / MCTS speedup).
    if (irreversible)
        next.hash_history.clear();
    next.hash_history.push_back(next.hash);
    return next;
}

// ============================================================
//  Game state detection
// ============================================================

bool Board::is_checkmate() const {
    return in_check() && legal_moves().empty();
}

bool Board::is_stalemate() const {
    return !in_check() && legal_moves().empty();
}

bool Board::is_fifty_move() const {
    return halfmove_clock >= 100;
}

bool Board::is_threefold() const {
    if (hash_history.size() < 5) return false;
    int count = 0;
    uint64_t h = hash;
    for (int i = (int)hash_history.size() - 1; i >= 0; i -= 2) {
        if (hash_history[i] == h) ++count;
        if (count >= 3) return true;
    }
    return false;
}

// Mirrors python-chess Board.has_insufficient_material(color) exactly so the
// C++ verdict agrees with the validator used in the parity tests.
static bool has_insufficient_material(const Board &b, int color) {
    int opp = 1 - color;
    Bitboard own  = b.occupied_by[color];
    Bitboard P = b.pieces[WHITE*6+PAWN]   | b.pieces[BLACK*6+PAWN];
    Bitboard N = b.pieces[WHITE*6+KNIGHT] | b.pieces[BLACK*6+KNIGHT];
    Bitboard Bp= b.pieces[WHITE*6+BISHOP] | b.pieces[BLACK*6+BISHOP];
    Bitboard R = b.pieces[WHITE*6+ROOK]   | b.pieces[BLACK*6+ROOK];
    Bitboard Q = b.pieces[WHITE*6+QUEEN]  | b.pieces[BLACK*6+QUEEN];
    Bitboard K = b.pieces[WHITE*6+KING]   | b.pieces[BLACK*6+KING];

    if (own & (P | R | Q)) return false;

    if (own & N) {
        // A lone knight is insufficient only if the opponent has nothing
        // (besides king/queens) that could help it get mated.
        return popcount(own) <= 2 &&
               !(b.occupied_by[opp] & ~K & ~Q);
    }

    if (own & Bp) {
        // Bishops insufficient iff all bishops (both sides) share a square
        // colour and there are no knights or pawns anywhere.
        const Bitboard DARK  = 0xAA55AA55AA55AA55ULL;
        const Bitboard LIGHT = ~DARK;
        bool same_color = !(Bp & DARK) || !(Bp & LIGHT);
        return same_color && !N && !P;
    }

    return true;
}

bool Board::is_insufficient_material() const {
    return has_insufficient_material(*this, WHITE) &&
           has_insufficient_material(*this, BLACK);
}

bool Board::is_draw() const {
    return is_fifty_move() || is_threefold() || is_insufficient_material();
}

// ============================================================
//  Piece count and phase
// ============================================================

int Board::piece_count() const {
    return popcount(occupied);
}

int Board::get_phase() const {
    int cnt = piece_count();
    if (cnt > 24) return PHASE_OPENING;
    if (cnt >= 12) return PHASE_MID;
    return PHASE_ENDGAME;
}

// ============================================================
//  Tensor encoding (21 planes, float32, shape 21×8×8)
// ============================================================

void Board::to_tensor(float *out) const {
    // out must point to 21*64 floats (allocated by caller)
    std::fill(out, out + 21 * 64, 0.0f);

    auto plane = [&](int p) -> float* { return out + p * 64; };

    // Planes 0-11: piece bitboards
    for (int p = 0; p < 12; ++p) {
        Bitboard bb = pieces[p];
        while (bb) {
            int sq = lsb(bb); bb &= bb - 1;
            plane(p)[sq] = 1.0f;
        }
    }

    // Plane 12: side to move (1.0 = white to move)
    if (side_to_move == WHITE)
        std::fill(plane(12), plane(12) + 64, 1.0f);

    // Plane 13: en passant target square
    if (ep_square != NO_SQ)
        plane(13)[ep_square] = 1.0f;

    // Planes 14-17: castling rights (broadcast)
    auto fill_plane = [&](int pl) { std::fill(plane(pl), plane(pl)+64, 1.0f); };
    if (castling & CASTLE_WK) fill_plane(14);
    if (castling & CASTLE_WQ) fill_plane(15);
    if (castling & CASTLE_BK) fill_plane(16);
    if (castling & CASTLE_BQ) fill_plane(17);

    // Plane 18: halfmove clock (normalised 0-1)
    float hmc = std::min(halfmove_clock, 100) / 100.0f;
    std::fill(plane(18), plane(18) + 64, hmc);

    // Plane 19: fullmove number (normalised, capped at 200)
    float fmn = std::min(fullmove_number, 200) / 200.0f;
    std::fill(plane(19), plane(19) + 64, fmn);

    // Plane 20: position already occurred (2-fold repetition marker)
    if (is_threefold() || [&]() {
            if (hash_history.size() < 3) return false;
            int cnt = 0;
            for (size_t i = 0; i < hash_history.size() - 1; i += 2)
                if (hash_history[i] == hash) ++cnt;
            return cnt >= 1;
        }())
        std::fill(plane(20), plane(20) + 64, 1.0f);
}

// ============================================================
//  Perft
// ============================================================

long long Board::perft(int depth) const {
    if (depth == 0) return 1LL;
    MoveList moves = legal_moves();
    if (depth == 1) return (long long)moves.size();
    long long nodes = 0;
    for (Move m : moves)
        nodes += apply_move(m).perft(depth - 1);
    return nodes;
}

// ============================================================
//  Null (pass) move — used for testing
// ============================================================

Board Board::null_move() const {
    Board next = *this;
    next.hash ^= ZOB_EP[(ep_square == NO_SQ) ? 8 : file_of(ep_square)];
    next.ep_square = NO_SQ;
    next.hash ^= ZOB_EP[8];
    next.side_to_move = 1 - side_to_move;
    next.hash ^= ZOB_BLACK_TO_MOVE;
    next.hash_history.push_back(next.hash);
    return next;
}

} // namespace chess
