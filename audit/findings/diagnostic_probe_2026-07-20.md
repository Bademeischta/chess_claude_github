# Diagnose-Probe des Netzes (audit/scripts/diagnostic_probe.py) — 2026-07-20

Getestet: `checkpoints/latest.pt` @ step 112381 (der laufende Post-Rollback-
Lauf). Stockfish (depth 18) als Schiedsrichter; KI-Suche batched, noise-frei.
Kernbefund: **scharfe, korrekte Bewertung bei konkret entschiedenen Stellungen
(Matt, sofort hängende Figur) — flache oder falsche Bewertung überall, wo
gerechnet oder Technik gebraucht wird.** Passt exakt zur objektiven ~440-Elo-Messung.

## Stärken (bestätigt)

- **Direkte Matts / 1-Zug-Material: absolut zuverlässig.** Ra8# (768/800 Visits,
  Eval +16), Rxd8# (255/256), exd5 (Dame nehmen), matcht jeweils SF. Bei einem
  konkreten Matt/Schlag konzentriert die Suche die Visits hart.
- **Value-Head gesund & ehrlich** in entschiedenen Stellungen: hohe Konfidenz,
  richtiges Vorzeichen. Kein Draw-Collapse.

## Schwächen (bestätigt, konkret)

- **Mehrzügige Taktik mit ruhigem Materialgewinn: blind.** Springergabel
  `4r1k1/7p/8/3N4/8/8/6PP/6K1 w`: Nf6+ gewinnt die Qualität (SF +4.4). Die KI
  spielt bei 256/1024/**4096** Sims durchgehend das passive Kh1 und bewertet
  sich mit **−1.5** (glaubt, verloren zu stehen). Mehr Suche macht es SCHLECHTER
  (Kh1-Visits 55→308→1824). Der Value-Head bewertet die materiell-unterlegene
  Stellung statisch und sieht die Gabel nie, die das Material dreht.
- **Endspieltechnik: kann K+T gegen K nicht verwandeln.** Weiß um den Gewinn
  (+16), aber im Ausspielen gegen Stockfish: Turm pendelt a4–h4 sinnlos,
  schwarzer König nie an den Rand gedrängt → **Remis nach 45 Halbzügen** aus
  `4k3/8/8/8/8/8/8/R3K3 w`. Keine Mattführung. (Endspiele tauchen im Self-Play
  kaum im Trainingssatz auf — deshalb ist endgame_curriculum_fraction relevant.)
- **Eröffnungs-Policy ist FLACH, nicht bloß „schlecht".** In Grundstellung /
  nach 1.e4 / 1.d4 liegen die Top-3-Züge bei ~identischen Visits (z.B.
  c4=130/e4=129/d4=129; d6=129/c5=128/d5=128). Die Policy hat kaum Präferenz —
  meist kommt ein vernünftiger Zug raus (c4, d6, d5), aber mit Zufall an der
  Spitze. Das versöhnt das Partie-`1…h5` (Münzwurf → Müll) mit dem Probe-`1…d6`
  (Münzwurf → ok): keine random Policy, sondern eine ohne starke Meinung.

## Null-Tests (ehrlich — Stellungsdesign meinerseits mangelhaft)

- **Unterverwandlung**: meine Stellung war ohnehin Remis (Dame gewinnen →
  K+S gegen K = remis), belohnte die Unterverwandlung also nicht. Untestbar.
  (Nebenbei: e8=D wurde korrekt über die verlierenden e8=T/e8=L gestellt.)
- **Zugzwang-Trébuchet**: Stellung war objektiv remis (nicht Weiß-verloren wie
  gedacht); die KI navigierte sauber (leicht pessimistische Eval, Remiszug).

## Synthese fürs Training

Stark genug, um Patzer sofort zu bestrafen; hilflos bei allem, was Plan,
Rechnung oder Technik braucht. Konkrete Hebel: (1) Endgame-Curriculum /
Distillation, die Mattführung *vorzeigt*; (2) schärfere Policy durch mehr/
bessere Daten (die flache Eröffnungs-Policy sollte mit Distillation anziehen).
