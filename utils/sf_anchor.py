"""
utils/sf_anchor.py — objective strength probe against a FIXED Stockfish anchor.

The opponent pool's internal Elo (training/pbt.py) is self-referential: it
rates the net against its own past selves, so systematic drift (both nets
sharing a blind spot, draw-heavy arenas) moves the whole scale without the
net getting stronger against the outside world. This module provides the
missing external yardstick: a short match against Stockfish pinned to a
known strength, reusable by

  * main.py            — automatic probe every `sf_elo_every_n_steps`
                         training steps, logged to TensorBoard
                         (`elo/stockfish`) + checkpoints/sf_elo_history.csv
  * tools/elo_vs_stockfish.py — the interactive/manual measurement tool

Match protocol (mirrors the fixed-anchor path of elo_vs_stockfish.py,
verified in the 2026-06/07 audits): alternate colours, greedy noise-free
MCTS with subtree+history reuse via tree.advance(), python-chess referees
(claim_draw=True), Wald CI on the score → Elo difference vs the anchor.

Windows note: any caller printing this module's output through .bat or
redirection must have reconfigured stdout to UTF-8 (project convention).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional

# Rough Elo equivalents of Stockfish "Skill Level" 0-20 (community
# calibration; SF's own UCI_Elo floor is 1320, skill levels reach lower).
# Shared with tools/elo_vs_stockfish.py — keep in sync.
SKILL_ELO = [800, 950, 1100, 1250, 1400, 1500, 1600, 1700, 1800, 1900,
             2000, 2050, 2100, 2150, 2200, 2300, 2400, 2500, 2700,
             2850, 3000]


def resolve_stockfish_path(cfg) -> Optional[str]:
    """Best-effort Stockfish binary discovery: config path, then the
    auto-install sentinel written by main._maybe_install_stockfish."""
    cand = getattr(cfg, "stockfish_path", "") or ""
    if cand and Path(cand).exists():
        return str(Path(cand).resolve())
    sentinel = Path(__file__).resolve().parent.parent / "runs" / ".stockfish_installed"
    if sentinel.exists():
        cand = sentinel.read_text(encoding="utf-8").strip()
        if cand and cand != "FAILED" and Path(cand).exists():
            return str(Path(cand).resolve())
    return None


def play_anchor_match(
    model,
    device,
    cfg,
    *,
    games: int = 30,
    sf_skill: Optional[int] = 3,
    sf_elo: Optional[int] = None,
    sims: int = 0,
    movetime: float = 0.0,
    stockfish_path: Optional[str] = None,
    batched_bs: int = 32,
    engine_inf=None,
    sf_threads: int = 2,
    sf_hash_mb: int = 128,
    progress: Optional[Callable[[int, int, int, int, int], None]] = None,
) -> Optional[dict]:
    """Play a fixed-anchor match vs Stockfish; return a result dict or None.

    None means "could not measure" (no Stockfish binary / engine failure) —
    callers treat that as a skipped probe, never as an error.

    sf_skill: Stockfish Skill Level 0-20 (anchor Elo from SKILL_ELO).
              Use for opponents below SF's UCI_Elo floor of 1320.
    sf_elo:   UCI_LimitStrength anchor (1320-3190) — overrides sf_skill
              when given.
    sims:     our MCTS sims/move (0 → cfg.mcts_sims).
    movetime: SF think time per move in seconds (0 → fair-time auto:
              max(0.05, sims * 0.000125), the measured ~0.125 ms/sim of the
              batched search — audit P7-2 calibration).
    batched_bs: leaves per shared NN forward (run_batched_sims); 0 = the
              slow single-leaf loop.
    engine_inf: optional shared inference engine (else each game's tree
              builds a TorchInferenceEngine over `model`).

    Result dict keys: wins, draws, losses, games, anchor_elo, anchor_label,
    elo_diff, ci_lo, ci_hi, estimated_elo, est_lo, est_hi, seconds.
    """
    try:
        import chess
        import chess.engine
    except ImportError:
        return None

    from engine.board import Board
    from engine.movegen import _move_uci
    from mcts.tree import MCTSTree
    from utils.elo import ELOSystem

    sf_path = stockfish_path or resolve_stockfish_path(cfg)
    if not sf_path:
        return None

    n_sims = int(sims) if sims and sims > 0 else int(cfg.mcts_sims)
    if movetime <= 0.0:
        movetime = max(0.05, n_sims * 0.000125)

    try:
        sf = chess.engine.SimpleEngine.popen_uci(sf_path)
    except Exception:
        return None

    t0 = time.time()
    was_training = getattr(model, "training", False)
    try:
        try:
            sf.configure({"Hash": int(sf_hash_mb), "Threads": int(sf_threads)})
        except chess.engine.EngineError:
            pass

        if sf_elo is not None:
            anchor = int(sf_elo)
            label = f"SF@UCI_Elo {anchor}"
            try:
                sf.configure({"UCI_LimitStrength": True, "UCI_Elo": anchor})
            except chess.engine.EngineError:
                sf.quit()
                return None
        else:
            skill = max(0, min(20, int(sf_skill if sf_skill is not None else 3)))
            anchor = SKILL_ELO[skill]
            label = f"SF skill {skill} (~{anchor})"
            try:
                # UCI_LimitStrength must be OFF or SF ignores Skill Level.
                sf.configure({"UCI_LimitStrength": False, "Skill Level": skill})
            except chess.engine.EngineError:
                sf.quit()
                return None

        model.eval()
        wins = draws = losses = 0
        for gi in range(int(games)):
            we_white = (gi % 2 == 0)
            pyb = chess.Board()
            ob = Board.from_fen(pyb.fen())
            tree = MCTSTree(cfg, model, device, engine=engine_inf)
            tree.reset(ob)

            while not pyb.is_game_over(claim_draw=True):
                legal = ob.legal_moves()
                uci_map = {_move_uci(m): m for m in legal}
                our_turn = (pyb.turn == chess.WHITE) == we_white
                if our_turn:
                    if batched_bs > 0:
                        done = 0
                        while done < n_sims:
                            adv = tree.run_batched_sims(
                                n_sims - done, batch_size=batched_bs)
                            if adv <= 0:
                                break
                            done += adv
                    else:
                        for _ in range(n_sims):
                            tree.run_one_simulation()
                    a = tree.root_analysis(top_k=1, pv_len=1)
                    move_int = (a["top"][0][0] if a["top"]
                                else tree.select_move(0.0)[0])
                    if move_int == 0:
                        break  # degenerate root — referee scores it below
                    pyb.push(chess.Move.from_uci(_move_uci(move_int)))
                else:
                    res = sf.play(pyb, chess.engine.Limit(time=float(movetime)))
                    pyb.push(res.move)
                    move_int = uci_map.get(res.move.uci())

                # Subtree + GRU-history reuse (the per-ply reset of the old
                # tool measured the net on out-of-distribution zero-history
                # inputs — audit P7-2).
                if move_int:
                    ob = ob.apply_move(move_int)
                    tree.advance(move_int, ob)
                else:
                    ob = Board.from_fen(pyb.fen())
                    tree.reset(ob, keep_history=True)

            res = pyb.result(claim_draw=True)
            if res == "1/2-1/2" or res == "*":
                draws += 1
            elif (res == "1-0") == we_white:
                wins += 1
            else:
                losses += 1
            if progress is not None:
                try:
                    progress(gi + 1, int(games), wins, draws, losses)
                except Exception:
                    pass
    finally:
        try:
            sf.quit()
        except Exception:
            pass
        if was_training:
            model.train()

    diff, lo, hi = ELOSystem.elo_difference_ci(wins, draws, losses)
    # Degenerate series (all wins / all losses) give a mathematically
    # unbounded Elo difference (score→0 ⇒ diff→−∞; the CI helper caps it
    # at ±2400, which still wrecks TensorBoard plot scaling and reads as
    # data, not as what it is: a BOUND). Clamp to the ±400 convention and
    # flag it — callers should adjust the anchor when this fires.
    bound_only = (wins + draws == 0) or (losses + draws == 0)
    clamp = lambda x: max(-400.0, min(400.0, x))  # noqa: E731
    diff, lo, hi = clamp(diff), clamp(lo), clamp(hi)
    return {
        "wins": wins, "draws": draws, "losses": losses,
        "games": wins + draws + losses,
        "anchor_elo": anchor, "anchor_label": label,
        "elo_diff": diff, "ci_lo": lo, "ci_hi": hi,
        "estimated_elo": anchor + diff,
        "est_lo": anchor + lo, "est_hi": anchor + hi,
        "bound_only": bound_only,
        "seconds": time.time() - t0,
    }
