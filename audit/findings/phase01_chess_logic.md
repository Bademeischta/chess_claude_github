# Phase 1 — Chess Logic Correctness (static analysis; runtime verification pending)

Files audited line-by-line: engine/board.py, engine/movegen.py, engine/rules.py, engine/chess_core.h, engine/chess_core.cpp (1119 ln, full read).

## 1.1 Library identification

- Primary: custom C++ engine (`chess_ext.pyd`, magic bitboards, Zobrist, full legal movegen) via pybind11.
- Fallback: python-chess 1.11.2 (board.py else-branch), activated when import fails or `CHESS_AI_NO_CPP=1`.
- python-chess also used by C++ path for: mate/stalemate disambiguation in rules.get_game_result (crash workaround), parity tests, Syzygy probing.

## 1.2 Board representation

- C++: 12 bitboards `pieces[color*6+type]`, occupied/occupied_by, a1=0…h8=63, rank-major (`rank_of(sq)=sq>>3`). Tensor row 0 = rank 1. White-POV always; **no flipping for Black** — plane 12 (all-1s when White to move) disambiguates.
- Zobrist: pieces, castling (full 4-bit index), EP file (9 entries incl. none), side. EP square set **only when an enemy pawn is positioned to capture** (chess_core.cpp:902-909) — matches python-chess "legal" FEN convention in spirit, BUT:

### FINDING P1-1 (LOW, correctness corner): pseudo-legal vs legal EP gate
chess_core.cpp:904 `if (PAWN_ATTACKS[us][ep] & next.pieces[them*6+PAWN])` checks an adjacent enemy pawn EXISTS, not that the EP capture is LEGAL (pawn may be pinned). python-chess `fen()` default `en_passant="legal"` emits the EP square only when a fully legal EP capture exists. Consequence: rare divergence in (a) parity-test fen4 comparison, (b) Zobrist key → threefold detection (position counted as distinct when EP capturer is pinned), (c) NN plane 13. Verify with a crafted pinned-EP FEN in runtime tests.

### FINDING P1-2 (MEDIUM, parity bug): plane 20 (repetition marker)
- C++ chess_core.cpp:1080-1087 sets plane 20 when position occurred before. The fallback lambda iterates `for (i=0; i<hash_history.size()-1; i+=2)` — **only even indices**. Prior same-side occurrences of the current position live at indices `size-1-2k`; when `size-1` is odd those are all odd indices → **marker missed in ~half of all cases** (depends on history-length parity).
- Python fallback board.py to_tensor **never writes plane 20 at all** (lines 247-269) → C++/Python tensor parity broken for any repeated position.
- Impact: NN input feature is noisy/unreliable; net cannot reliably learn "this is a repetition → draw-ish". Affects draw handling quality, not legality.
- Fix: count occurrences of `hash` in `hash_history[0..size-2]` at ANY index with matching parity stepping from `size-1-2` downward by 2 (or simply compare all entries — hash includes side-to-move so cross-parity matches are impossible anyway); implement plane 20 in the Python fallback too.

### FINDING P1-3 (LOW): stale-move recovery path returns corrupted hash
chess_core.cpp:831-844: when `moving < 0` (stale move), function returns `next` AFTER `next.hash ^= ZOB_EP[...]` and `next.hash ^= ZOB_CASTLING[...]` were already applied but never re-added → returned board's Zobrist desyncs from its position. Poisons repetition detection for that subtree. Should never trigger (diagnostic counter); fix by returning a clean copy of `*this`.

## 1.3/1.4/1.5 Move generation (C++, verified by reading)

- Pawn: single/double push (rank guards), promotions on rank 6→7 push & captures (add_promos emits all 4 = N,B,R,Q), EP generation from PAWN_ATTACKS ∩ ep_square. Promotion-capture victim removal verified present (chess_core.cpp:851-858 — comment documents this exact past bug).
- Castling (chess_core.cpp:761-789): rights bit + empty squares (F1G1 / B1C1D1) + E1,F1,G1 (resp. E1,D1,C1) not attacked. B1 attacked correctly NOT checked (only rook passes through b1 — legal). Rights updated via CASTLING_MASK[from] & CASTLING_MASK[to] (covers rook capture on its home square). Correct per FIDE.
- Legality filter: apply each pseudo move, reject if own king attacked (lines 626-641). Covers pins, EP-discovered-check, moving into check. Correct but O(moves × board-copy) — perf measured in 1.7.
- is_threefold (969-978): counts current hash at same-parity indices, ≥3 → draw; history CLEARED on irreversible moves (correct: repetition cannot span pawn move/capture) — also keeps Board copies cheap.
- Fifty-move: halfmove_clock ≥ 100, reset on pawn move/capture/promo (926-931). Correct.
- Insufficient material (982-1016): mirrors python-chess both-sides semantics incl. same-color-bishops. Cross-checked by parity suite.
- Perft suite in movegen.py: 6 standard CPW positions incl. Kiwipete + promo-heavy positions; parity suite: 8 fixed endgame FENs + 200 random games × 60 plies vs python-chess. Good coverage; runtime results recorded below.

## 1.6 Move encoding (movegen.py)

- 64×73=4672 AlphaZero layout: 56 queen-ray channels + 8 knight + 9 underpromo. Queen-promo encoded as normal queen move (standard).
- **Black perspective flip is `63 - sq` = 180° rotation, NOT vertical mirror `sq ^ 56`** (movegen.py:78-80). Board tensor is NEVER flipped. See FINDING P1-4.
- Underpromotion dir_idx from flipped df: self-consistent for both colors (decode is brute-force re-encode over legal moves → bijective within a position by construction; collision test still scheduled in runtime verification).
- `move_to_action_index` is @lru_cache(maxsize=None) — pure function, sound; unbounded but domain ≤ 2^16×2.

### FINDING P1-4 (HIGH, design): hybrid perspective convention
Board input: always White-POV (+ side-to-move plane 12). Policy output: action indices 180°-rotated when Black to move. The network must learn TWO action-space semantics gated by plane 12, while convolution weights see un-flipped boards for both sides. AlphaZero/Lc0 instead flip the BOARD so weights are color-equivariant. Consequences: (a) ~halves effective sample efficiency for the policy head; (b) any consumer that forgets the flip produces garbage — consistency across producers/consumers verified so far: `policy_array_to_legal_priors(…, side_to_move)` (MCTS priors), `select_move` target building (tree.py:616-622) — both use the same convention. Remaining consumers to check: trainer (P5), web GUI/UCI (P7). Not a bug per se; flagged as architecture deviation (P9) with measurable cost.

### FINDING P1-5 (MEDIUM): `action_index_to_move` returns 0 sentinel
movegen.py:123-131 returns **0** when index unmatched; 0 decodes as a1→a1 quiet "move". C++ apply_move(0) on a board with a white piece on a1 acts as a PASS (remove+put same square, side flips!) — does NOT hit the stale-move guard. Callers must never apply it. Caller audit: tree.py select_move fallback path can return move 0 when root has no children AND no legal moves (tree.py:598) — game_stream then calls board.apply_move(0) (tree.py:1083) before get_game_result. Reachable when a random-opening (`random_opening_plies`) game starts in a terminal position (e.g. fool's-mate within ≤6 random plies) — see FINDING P4-2 in phase04 file.

## 1.x Python fallback specific

### FINDING P1-6 (LOW): fallback `hash` property likely AttributeError
board.py:327 `return self._b.zobrist_hash()` — python-chess `Board` has no `zobrist_hash()` method (zobrist hashing lives in `chess.polyglot.zobrist_hash(board)`). Any caller touching `.hash` in fallback mode crashes. Runtime-verify (1 line) then file as bug.
### FINDING P1-7 (PERF): fallback apply_move rebuilds via FEN and copies full FEN-string history per move — O(game²) chars; plus python-chess board reparse per move. Fallback is 'correct but slow' as documented; quantify in 1.7 benchmark.
### Fallback is_draw does not include stalemate — consistent with C++ (stalemate handled as terminal-no-moves separately in rules.get_game_result). Correct.

## Runtime verification & benchmarks: TODO (after static phase): pytest tests/test_board.py, main.py --test, verify_chess_logic.py (20-moves check, EP/castle/underpromo FENs, encode-bijectivity over random games, pinned-EP probe, plane-20 repro), movegen perf benchmark.

## SELF-EXPAND tasks
- T1.1 Build encode-bijectivity test incl. Black 180° rotation and underpromotions both colors.
- T1.2 Craft pinned-EP FEN; compare C++ ep_square/FEN vs python-chess (P1-1).
- T1.3 Repro plane-20 parity bug with a 4-fold shuffle sequence; verify both parities (P1-2).
- T1.4 Verify fallback `.hash` AttributeError (P1-6).
- T1.5 Confirm stale_move_count == 0 after self-play window (ties to P1-5/P4-2).
- T1.6 Verify .pyd binary behavior matches current .cpp source (T0.1) — e.g. plane-20 lambda present in binary? Test via behavior, source diff vs git log of .cpp files.
