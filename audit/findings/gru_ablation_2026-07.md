# GRU-Ablation (audit S13) — 2026-07-16

Zwei from-scratch-Läufe, identische Seeds (20260715) und Config bis auf
`use_gru_history`, je 4.0 h Wallclock, 50 Sims/Zug
(Bootstrap-Regime), danach SF-Anker (Skill 0, 40 Partien,
128 Sims) und 60-Partien-Direktvergleich. Treiber:
audit/scripts/gru_ablation.py; Rohdaten: gru_ablation.json.

| Metrik | GRU an | GRU aus |
|---|---|---|
| Trainingssteps | 9280 | 9994 |
| Self-Play-Positionen | 2394748 | 2607978 |
| Self-Play pos/s | 185.8 | 198.5 |
| Median train-step (ms) | 155.8 | 128.5 |
| SF-Anker-Elo | 400 (CI 400..400, W0/D0/L40) | 400 (CI 400..531, W0/D7/L33) |

**Direktvergleich (nogru vs gru, 60 Partien):**
nogru W0 / D20 / gru W40

## Interpretation (gesichtet 2026-07-16)

**Entscheidung: GRU BLEIBT.** Die S13-Hypothese („GRU kostet ~12 % Latenz,
vermutlich ohne Stärkeverlust entfernbar") ist für das Bootstrap-Regime
widerlegt:

- Der Durchsatzgewinn ohne GRU ist real, aber klein: 1,07× pos/s
  (185,8 → 198,5) und 128,5 statt 155,8 ms/Trainstep (−17,5 %).
- Der **Direktvergleich ist vernichtend eindeutig**: GRU gewinnt 40/60,
  verliert 0/60 (Score 83 % ≈ +280 Elo) — trotz ~9 % weniger Daten und
  714 weniger Trainingssteps. Die 4 History-Kontext-Planes tragen im
  Self-Play-Bootstrap also substanziell.
- Die SF-Proben (Skill 0) sind bei beiden noch Bound-only (0 bzw. 7 Remis
  aus 40) — 4-h-from-scratch-Netze liegen unter ~800 Elo; als
  Diskriminator taugt hier nur der Direktvergleich. Kuriosum: nogru holte
  7 SF-Remis, gru 0 — bei n=40 und derart schwachen Netzen Rauschen.

Einschränkungen: gemessen im Bootstrap-Regime (50 Sims, 4 h, from
scratch, reines Self-Play). Für das Produktionsnetz mit Teacher-Daten
(durchweg Null-Historien — Distributionsbruch, S13-Argument) könnte die
Bilanz anders aussehen; ein Re-Test lohnt frühestens nach einem
Distillations-Zyklus mit History-tragenden Teacher-Daten. Bis dahin:
`use_gru_history=True` bleibt Default.
