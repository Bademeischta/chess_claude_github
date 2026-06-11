╔══════════════════════════════════════════════════════════════════════════════════╗
║         TOTAL SYSTEMS AUDIT — CHESS AI PROJECT (ALPHAZERO / PYTORCH / MCTS)     ║
║                     LONG-RUNNING DEEP ANALYSIS SESSION                           ║
╚══════════════════════════════════════════════════════════════════════════════════╝

This is not a code review. This is a forensic deconstruction.

You are a principal AI systems engineer, ML researcher, chess engine architect, and
performance optimization specialist. You have been given unrestricted access to a
complete chess AI project. Your task is to understand, audit, benchmark, stress-test,
and ultimately improve every single aspect of this system.

This session has no time limit. You do not move on until a phase is truly complete.
You do not summarize when you can measure. You do not assume when you can verify.
You do not skip a detail because it seems minor — in an ML self-play system, minor
bugs compound over millions of iterations into catastrophic failures that are nearly
impossible to diagnose after the fact.

═══════════════════════════════════════════════════════════
TARGET HARDWARE PROFILE
═══════════════════════════════════════════════════════════

GPU:             NVIDIA RTX 5070
VRAM:            12 GB GDDR7
Architecture:    Blackwell (GB205)
CUDA Cores:      ~6144
Tensor Cores:    Yes (5th gen, FP8/FP16/BF16/TF32)
Max Bandwidth:   ~896 GB/s
NVLink:          No (single GPU setup)
CPU:             Unknown — audit and document
RAM:             Unknown — audit and document
Storage:         Unknown — audit and document (matters for replay buffer I/O)
OS:              Unknown — audit and document
CUDA version:    Unknown — audit and verify compatibility
PyTorch version: Unknown — audit and verify

All performance estimates and recommendations must be grounded in this hardware profile.
When you don't know a spec, note it and factor in a conservative estimate.

═══════════════════════════════════════════════════════════
PROJECT PROFILE
═══════════════════════════════════════════════════════════

Style:      AlphaZero (DeepMind, 2017/2018)
Search:     Monte Carlo Tree Search (MCTS) with neural network guidance
Training:   Self-play reinforcement learning
Framework:  Python + PyTorch
Target:     Chess (standard rules, likely standard piece representation)

═══════════════════════════════════════════════════════════
OPERATING PROCEDURE — READ THIS FIRST AND FOLLOW IT ALWAYS
═══════════════════════════════════════════════════════════

TASK QUEUE PROTOCOL:
You maintain a living TASK QUEUE at all times, formatted as:

  [ ] = not started
  [~] = in progress
  [✓] = completed
  [!] = completed with critical finding
  [➕] = self-generated task added mid-audit

The queue starts with all phases below. It MUST grow before it shrinks. Every time
you encounter something suspicious, something you don't fully understand, something
that touches another part of the system, or something that raises a "wait, does this
mean X also has a bug?" question — you add a task. Immediately. Right then.

VERIFICATION PROTOCOL:
- Never write "this looks correct" without quoting the exact code and explaining why.
- Never write "performance seems fine" without a measurement or calculation.
- Never write "probably not an issue" — either investigate it or add it to the queue.
- If you catch yourself about to skip something, write [SKIPPING: reason] visibly,
  so it can be challenged.

DEPTH PROTOCOL:
When you find a bug or issue:
  1. Describe exactly what it is
  2. Quote the offending code
  3. Explain what the actual behavior is vs. intended behavior
  4. Trace all downstream effects — what else does this corrupt?
  5. Propose the exact fix with corrected code
  6. Add a task: "verify fix didn't break X, Y, Z"

When you find something working correctly:
  1. Explain why it's correct (don't just say "looks fine")
  2. Ask: is it correct but suboptimal? Add performance task if so.
  3. Ask: is it correct under all edge cases? List the edge cases you verified.

SELF-EXPANSION PROTOCOL:
At the end of every sub-phase, ask yourself:
  "What did I just learn that I didn't know before this phase?"
  "Does that knowledge change what I should check elsewhere?"
  "Is there a version of this bug that could exist in another module?"
  "What assumption am I making right now that I haven't verified?"

Add every answer that generates new work to the task queue.

═══════════════════════════════════════════════════════════
PHASE 0 — ORIENTATION & INITIAL SCAN
═══════════════════════════════════════════════════════════

Before touching any code in depth, perform a full orientation sweep:

□ 0.1  List every file in the project with its size, last modified date, and a 
       one-line description of its likely purpose
□ 0.2  Identify the primary entry points: training, self-play, evaluation, inference
□ 0.3  Count total lines of code by file and by category (model, search, training,
       utils, tests, scripts)
□ 0.4  List all third-party dependencies (requirements.txt / pyproject.toml / imports)
       — note exact versions where available
□ 0.5  Check if there are any README or documentation files — read them and note
       where they are outdated or incorrect
□ 0.6  Check git history if available: how many commits? What was the development
       trajectory? Any major rewrites? Any TODO comments left in code?
□ 0.7  Identify any configuration files (YAML, JSON, .env, argparse defaults) and
       document every configurable hyperparameter
□ 0.8  Check for test files — any unit tests, integration tests, or eval scripts?
       Run them if present, note results
□ 0.9  Check for any existing profiling data, training logs, loss curves, or
       checkpoint files — these are evidence of what has already run
□ 0.10 Identify potential red flags immediately visible from surface scan:
       - TODO/FIXME/HACK/BUG comments
       - Unusually large files (God modules)
       - Empty files
       - Files that seem misplaced
       - Duplicate logic in multiple places
       - Mixed indentation or formatting inconsistencies

SELF-EXPAND: After 0.10, add at least 5 specific follow-up tasks based on what
you found in the surface scan.

═══════════════════════════════════════════════════════════
PHASE 1 — CHESS LOGIC CORRECTNESS (Foundation Layer)
═══════════════════════════════════════════════════════════

CRITICAL: If chess logic is wrong, nothing else matters. Every game played will be
invalid. Training will converge on wrong policies. This phase must be airtight.

□ 1.1  LIBRARY IDENTIFICATION
       - Is python-chess used? If yes: which version? Are there known bugs in that version?
       - Is move generation custom-written? If yes: treat it as completely unverified.
       - Is there any move generation at all, or is it deferred to an external process?

□ 1.2  BOARD REPRESENTATION
       - What is the internal board representation? (bitboard, 8x8 array, 0x88, etc.)
       - Is there a consistent coordinate system? (rank 0 = rank 1 or rank 8? 
         a=0 or a=1?)
       - Is the board from White's perspective always, or does it flip for Black?
       - Is there a flip/mirror function for perspective-invariant training? Is it correct?
       - Check: does the same position always produce the same board tensor? (determinism)

□ 1.3  PIECE MOVE GENERATION — verify each piece type individually:
       PAWN:
         □ Forward movement (1 square, 2 squares from starting rank only)
         □ Diagonal captures
         □ En passant (correct square, correct removal of captured pawn,
           only valid for one ply — does the en passant flag reset correctly?)
         □ Promotion — all 4 promotion types? (Q, R, B, N)
         □ Promotion via capture — is this handled?
         □ Pawn cannot move backward under any circumstances
       KNIGHT:
         □ All 8 L-shapes, boundary handling
         □ Jumping over pieces (should be permitted)
       BISHOP:
         □ Diagonal rays, blocked by pieces
         □ Stays on its color square forever
       ROOK:
         □ Horizontal/vertical rays, blocked by pieces
         □ Castling rights flag updates on first rook move
       QUEEN:
         □ All 8 rays (bishop + rook combined)
       KING:
         □ All 8 adjacent squares
         □ Cannot move into check
         □ Castling: kingside AND queenside for BOTH colors (4 total)
           - King not currently in check
           - King does not pass through check
           - King does not land in check
           - Squares between king and rook are empty
           - Neither king nor rook has previously moved (flag tracking)
           - Rook on correct square
         □ Cannot capture a defended piece

□ 1.4  SPECIAL RULE CORRECTNESS
       □ Check detection — is it correctly identified in all cases?
         (including discovered check, double check)
       □ Checkmate detection — all legal moves exhausted while in check
       □ Stalemate detection — all legal moves exhausted while NOT in check
       □ Threefold repetition — is position history tracked? What counts as a 
         "position" (FEN without move counters? Full FEN?)
       □ Fifty-move rule — is the half-move clock tracked and reset correctly?
         (reset on pawn move or capture, not just captures)
       □ Insufficient material — is this detected? Which combinations?
         (K vs K, K+B vs K, K+N vs K, K+B vs K+B same color)
       □ Draw by agreement / immediate draw offers — not applicable to self-play,
         but check if there's any placeholder that could accidentally trigger

□ 1.5  LEGAL MOVE GENERATION CORRECTNESS
       □ Are moves filtered for check? (you cannot make a move that leaves your
         own king in check — this includes blocking discovered checks)
       □ Are all legal moves generated with no duplicates?
       □ Is the count of legal moves at the starting position exactly 20?
         (This is a known correct value and a useful sanity check)
       □ Run 10 random positions through legal move generation and verify manually
         against Stockfish or a known-good engine

□ 1.6  MOVE ENCODING & DECODING
       □ How are moves encoded for the policy head? (from/to squares? UCI string?
         custom encoding?)
       □ How many possible moves does the encoding support? (AlphaZero uses 4672
         for standard chess — verify this matches if similar encoding is used)
       □ Are underpromotions represented? (Knight, Bishop, Rook promotions —
         these are rare but legally important and occasionally the best move)
       □ Is there a bijective mapping between move encoding and legal moves?
         (no two different moves should map to the same index)
       □ What happens with moves that are in the policy output but illegal?
         (should be masked to -inf before softmax — verify this happens)
       □ Is the move encoding consistent between training data generation and
         inference/MCTS?

□ 1.7  PERFORMANCE BENCHMARK — MOVE GENERATION
       □ Measure: legal moves generated per second (single-threaded)
       □ Measure: time to generate all legal moves for starting position
       □ Measure: time to generate all legal moves for a complex middlegame position
         (use a famous position — e.g. Kasparov vs Deep Blue game 6, 1997)
       □ Compare: is this fast enough that move generation is NOT the bottleneck?
         If it is the bottleneck, flag as high priority performance issue.

SELF-EXPAND: What did you find in Phase 1 that needs follow-up elsewhere?
(Examples: if encoding is wrong → add task to verify MCTS uses same encoding;
if flip function is suspicious → add task to verify training data uses it correctly)

═══════════════════════════════════════════════════════════
PHASE 2 — NEURAL NETWORK ARCHITECTURE DEEP AUDIT
═══════════════════════════════════════════════════════════

□ 2.1  INPUT REPRESENTATION — THE BOARD ENCODING TENSOR
       Document the exact tensor format:
       □ Shape: (channels, 8, 8) — how many channels?
       □ For each channel, what does it represent? Common AlphaZero encoding has:
         - 6 channels × 2 colors = 12 piece-presence planes
         - 1 channel for side to move (all 0s or all 1s)
         - 4 channels for castling rights (KQ for white, kq for black)
         - 1 channel for en passant square (if applicable)
         - 1 channel for 50-move clock (normalized? how?)
         - Optional: T previous positions for repetition detection
       □ Is the board always encoded from the current player's perspective?
         (i.e., when it's Black's turn, does the board flip so Black's pieces are
         always at the bottom? This is the AlphaZero convention — verify it.)
       □ Are the values normalized to [0, 1]? Or raw integers?
       □ What dtype is the tensor? float32? float16? Does it match the model's dtype?
       □ Test: encode a known position, decode it back, verify it matches.
         Especially test positions with: en passant available, castling available,
         50-move clock > 0, Black to move (board should be flipped)

□ 2.2  RESIDUAL TOWER
       □ How many residual blocks?
       □ How many channels (filters)?
       □ Is batch normalization present? Before or after ReLU?
         (Original AlphaZero: Conv → BN → ReLU. Check this order.)
       □ Is the skip connection a simple addition or does it include a projection?
       □ Are there squeeze-and-excitation blocks? (Leela Chess Zero uses these)
       □ What activation function? ReLU everywhere? Any LeakyReLU, GELU, SiLU?
       □ Kernel sizes — 3×3 for residual, different for input/output heads?
       □ Is the model wrapped in nn.DataParallel or DDP? (not needed for single GPU
         but check anyway — DDP bugs can be silent)

□ 2.3  POLICY HEAD
       □ Full architecture of the policy head (layers, activation, output)
       □ Output shape: should match the move encoding space (e.g., 4672 for AlphaZero)
       □ Is there a softmax at the end?
         CRITICAL: There should NOT be a softmax if using CrossEntropyLoss
         (PyTorch's CrossEntropyLoss includes LogSoftmax internally — double softmax
         is a silent catastrophic bug. Verify this carefully.)
       □ During MCTS: are raw logits used, or probabilities? Which is more numerically
         stable for the PUCT formula?
       □ Is illegal move masking applied before or after softmax?
         (Must be before — set illegal move logits to -1e9, then softmax)

□ 2.4  VALUE HEAD
       □ Full architecture of the value head
       □ Output shape: scalar (1,) or 3-class (win/draw/loss)?
       □ Output activation: tanh (→ [-1, 1]) or sigmoid (→ [0, 1]) or none?
       □ What loss function is used? MSE? Cross-entropy for WDL?
       □ Is the value in [-1, 1] consistently? What represents win/loss/draw?
         (+1 = current player wins? Or +1 = White wins always? Document this.)
       □ CRITICAL: Is the value correctly negated when backing up through the MCTS tree?
         (After each node, the value flips sign because it's now the opponent's turn)

□ 2.5  MODEL PARAMETER AUDIT
       □ Count total parameters (trainable and non-trainable)
       □ Estimate model size in MB (float32: params × 4 bytes)
       □ Check: does it fit comfortably in RTX 5070 VRAM (12GB) with a large batch?
       □ Estimate VRAM usage during training: model + gradients + optimizer state
         (Adam optimizer stores 2 moment tensors → ~3× model size for optimizer state)
       □ Are any layers frozen that shouldn't be? Or trainable that shouldn't be?
       □ Are there any unusually large layers? (Could indicate architectural mistakes)

□ 2.6  WEIGHT INITIALIZATION
       □ Is weight initialization explicitly set, or relying on PyTorch defaults?
       □ For Conv2d: PyTorch default is Kaiming Uniform (good for ReLU)
       □ For Linear: PyTorch default is Kaiming Uniform
       □ For BatchNorm: weight=1, bias=0 (default — verify)
       □ Is there any orthogonal or spectral initialization? (Sometimes used for
         better gradient flow in deep networks)
       □ Check: does the value head produce near-zero outputs at initialization?
         (It should — predicting 0 (draw) at start is better than random ±1)

□ 2.7  FORWARD PASS CORRECTNESS
       □ Trace a single board position through the entire forward pass manually
       □ Verify output shapes at every layer
       □ Check: does the model handle batch size 1 correctly? (Important for MCTS
         single-position inference)
       □ Check: does the model handle batch size N correctly? (Important for training)
       □ Run a forward pass in both train() and eval() mode — do outputs differ
         significantly? (They should differ slightly due to BatchNorm/Dropout —
         but a large difference indicates a bug)
       □ Check gradient flow: run a backward pass with a dummy loss, verify all
         parameters receive gradients

SELF-EXPAND: List 5 new tasks generated from Phase 2 findings.

═══════════════════════════════════════════════════════════
PHASE 3 — MCTS IMPLEMENTATION FORENSIC AUDIT
═══════════════════════════════════════════════════════════

MCTS is the most complex component and the most common source of subtle bugs.
Every detail matters. Work through this with extreme care.

□ 3.1  TREE DATA STRUCTURE
       □ How is the tree represented in memory? (dict of nodes? objects? arrays?)
       □ What does a node store?
         - Visit count N(s,a)
         - Total action value W(s,a)
         - Mean action value Q(s,a) = W/N
         - Prior probability P(s,a) (from neural network)
         - Child nodes
         - Parent node reference (needed for backprop if iterative)
         - Board state (stored? reconstructed on demand?)
       □ Is the tree persistent across moves (tree reuse) or rebuilt each move?
         If tree reuse: is the subtree correctly extracted and re-rooted?
       □ Check memory usage: for 800 simulations with branching factor ~30,
         the tree has ~24,000 nodes — is memory managed correctly?
       □ Is there a maximum tree size limit? What happens when it's hit?

□ 3.2  SELECTION PHASE (UCB/PUCT)
       The PUCT formula from AlphaZero:
         U(s,a) = Q(s,a) + c_puct × P(s,a) × sqrt(N(s)) / (1 + N(s,a))
       
       □ Is this formula implemented exactly? Quote the code.
       □ What is c_puct? AlphaZero uses ~1.0. Leela uses ~2.5+. Is it a constant
         or does it scale? Is it tuned or just a default guess?
       □ Numerical stability at N(s,a) = 0: does it handle divide-by-zero?
       □ Numerical stability at N(s) = 0 (root node first visit): what happens?
       □ Is there a tie-breaking mechanism when multiple children have equal UCB?
         (Random tie-breaking is better than always picking the first)
       □ Check: the selection recurses down the tree until it hits an unexpanded node
         OR a terminal node. Is this correctly implemented?
       □ For terminal nodes: is the correct terminal value returned immediately?
         (No network evaluation needed for checkmate/stalemate/draw)
       □ Are terminal nodes correctly identified? (Not just checkmate — all draw types)

□ 3.3  EXPANSION PHASE
       □ When a leaf node is selected for the first time, it must be expanded:
         - Generate all legal moves
         - Evaluate position with neural network → get (policy, value)
         - Initialize child nodes with priors from policy (with illegal move masking)
         - Store value for backpropagation
       □ CRITICAL: Are illegal moves masked BEFORE initializing priors?
         (Priors for illegal moves should be 0, not renormalized softmax values)
       □ After masking, are the remaining priors renormalized to sum to 1?
       □ If the position is terminal (just discovered during expansion), is the
         network evaluation skipped and the correct game result returned?
       □ Is the newly expanded node immediately backed up? (It should be)

□ 3.4  BACKUP PHASE
       □ Is value backed up correctly through the full path to root?
       □ Is the value negated at each alternating level?
         (V for current player = -V for opponent = V for current player's grandparent)
       □ Is N(s,a) incremented for every visited node/edge on the path?
       □ Is W(s,a) updated atomically? (Matters for parallel MCTS)
       □ Is Q(s,a) computed as W/N after each update? Or lazily?

□ 3.5  DIRICHLET NOISE
       □ Is Dirichlet noise added at the ROOT node only?
         (Adding it to non-root nodes is a known bug — it breaks the search)
       □ What alpha value is used? AlphaZero uses α = 0.3 for chess.
         Appropriate values depend on branching factor: α = 10 / avg_moves
         At ~30 average legal moves: α ≈ 0.33 (close to 0.3)
       □ What epsilon is used? AlphaZero uses ε = 0.25 (25% noise, 75% prior)
       □ Is noise only added at the start of each search, not re-added per simulation?
       □ Is noise only added during TRAINING self-play, NOT during evaluation games?
         (Noise during evaluation makes results non-deterministic and weakens play)

□ 3.6  TEMPERATURE
       □ What temperature is used for move selection after MCTS finishes?
       □ τ = 1.0 for the first N moves (encourages exploration during training)
       □ τ → 0 (argmax) after N moves (sharper play in late game)
       □ What is N? AlphaZero uses 30 for chess.
       □ Is temperature correctly applied to visit counts (not Q values)?
         Move probability ∝ N(s,a)^(1/τ)
       □ Is temperature set to 0 (argmax) during evaluation games?

□ 3.7  VIRTUAL LOSS (if parallel MCTS is implemented)
       □ Is virtual loss implemented?
       □ Virtual loss adds a penalty to discourage multiple threads from visiting
         the same node simultaneously
       □ Is it correctly removed after backup?
       □ Is the virtual loss magnitude reasonable? (Should be similar scale to
         actual value estimates, typically 1.0 or 3.0)
       □ Are there race conditions in the tree? (Multiple threads expanding same node?)
       □ If not parallel: note this as a potential future optimization

□ 3.8  TRAINING DATA GENERATION FROM MCTS
       □ After a game completes, what data is stored?
         - Board states (as input tensors)
         - MCTS policy targets (visit counts normalized to probabilities)
         - Game outcome (value targets: +1, 0, -1 from each position's perspective)
       □ Is the value target consistently from the CURRENT PLAYER'S perspective?
       □ Is the policy target the FULL visit distribution, or just the chosen move?
         (Full distribution is the correct AlphaZero approach)
       □ Are BOTH players' positions stored from each game? (They should be)
       □ Is there any data augmentation? (Horizontal flip is valid for chess if
         both the board AND the move encoding are flipped consistently)
       □ CRITICAL: If data augmentation is used, are move encodings also transformed?
         A horizontally flipped board must have horizontally flipped move indices too.

□ 3.9  MCTS PERFORMANCE BENCHMARK
       □ Measure: simulations per second (single-threaded, no batching)
       □ Measure: simulations per second (with batched inference, if implemented)
       □ Identify: where is the time spent?
         - Tree traversal / selection: ___ms per sim
         - Neural network inference: ___ms per sim
         - Move generation: ___ms per sim
         - Memory allocation for new nodes: ___ms per sim
       □ At 800 simulations, what is the total time per move?
       □ How does it compare to AlphaZero's reported 800 simulations in ~30ms?
       □ Is inference the bottleneck? (If yes → batching or GPU optimization needed)
       □ Is Python overhead the bottleneck? (If yes → numba, C extension, or 
         architectural changes needed)

SELF-EXPAND: What did Phase 3 reveal about the interaction between MCTS and
the neural network that needs further investigation?

═══════════════════════════════════════════════════════════
PHASE 4 — SELF-PLAY PIPELINE COMPLETE AUDIT
═══════════════════════════════════════════════════════════

□ 4.1  GAME LOOP STRUCTURE
       □ Trace one complete game from board initialization to game-over storage.
         Document every step in sequence.
       □ Is there a maximum game length? AlphaZero limits to 512 moves.
         What happens if this limit is hit? Is it stored as a draw?
       □ Is the game result correctly propagated backward to all stored positions?
       □ Is the game loop deterministic? (Given a fixed random seed, should produce
         identical games — useful for debugging)
       □ Check: is the board correctly reset between games?
       □ Check: is the MCTS tree correctly reset between games? (Or does it carry
         over state from the previous game — this would be a bug)

□ 4.2  REPLAY BUFFER
       □ What data structure is the replay buffer? (deque? list? numpy array? 
         memory-mapped file?)
       □ Maximum size: how many positions? At roughly 40 positions per game, 
         how many games does this correspond to?
       □ Is it FIFO (oldest discarded when full)? Or random replacement? Or other?
       □ Is the buffer on CPU or GPU? (Should be CPU to save VRAM)
       □ How is sampling performed? Uniform random? Prioritized?
       □ Is there a minimum fill threshold before training begins?
         (Training on a near-empty buffer leads to severe overfitting)
       □ Is the buffer shared between self-play and training? (Threading/async issues?)
       □ VRAM-CPU bandwidth: when a training batch is sampled, how is it moved to GPU?
         (pin_memory=True + non_blocking=True for async transfer)
       □ Memory estimate: 1M positions × (channels × 8 × 8 × 4 bytes + 4672 × 4 bytes
         + 4 bytes) ≈ how many GB? Does this fit in RAM?

□ 4.3  SELF-PLAY PARALLELIZATION
       □ Are multiple games played simultaneously? How many workers?
       □ Method: multiprocessing? threading? asyncio? Ray? None?
       □ If multiprocessing: how is the neural network shared?
         (Can't share PyTorch model directly across processes without careful handling)
       □ If threading: is the GIL a problem? (It is for CPU-bound work)
       □ If no parallelization: flag this as a significant throughput limitation.
         GPU sits idle during tree traversal — this is a major inefficiency.
       □ Check for race conditions in the replay buffer if multi-process self-play
         writes to it concurrently
       □ Check for race conditions in the MCTS tree if multi-threaded per game

□ 4.4  TRAINING / SELF-PLAY PIPELINE COORDINATION
       □ Is self-play and training done sequentially or concurrently?
         (Sequential: generate N games, then train N steps, repeat — simple but slow)
         (Concurrent/async: self-play workers fill buffer while training runs — 
          much faster but more complex)
       □ If concurrent: how is the model updated in self-play workers?
         (Need periodic model sync to self-play workers — is this implemented?)
       □ What triggers a model update? After N training steps? After N games?
         After ELO gain threshold?
       □ AlphaZero architecture: self-play always uses latest checkpoint.
         Leela Chess Zero architecture: self-play uses best performing model.
         Which approach is used here? Is it consistently implemented?

□ 4.5  OPENING DIVERSITY
       □ Is there a mechanism to ensure diverse opening play?
         (Without it, self-play games converge to the same opening repeatedly)
       □ Dirichlet noise at root helps — is it sufficient here?
       □ Is there an opening book? Pros: faster early game exploration. Cons: 
         might bias learning toward specific openings.

□ 4.6  ESTIMATED THROUGHPUT
       □ Calculate: games per hour on target hardware
         = (seconds per game) / 3600
         Where: seconds per game ≈ avg_moves × simulations_per_move × time_per_sim
       □ Calculate: training steps per hour
       □ Calculate: estimated time to generate 1M training positions
       □ Compare to AlphaZero paper: they used 5000 TPUv1s. This project uses one GPU.
         What is the realistic training time estimate for a meaningful result?

SELF-EXPAND: Pipeline bottlenecks → add performance optimization tasks.

═══════════════════════════════════════════════════════════
PHASE 5 — TRAINING LOOP & OPTIMIZATION DEEP AUDIT
═══════════════════════════════════════════════════════════

□ 5.1  LOSS FUNCTION ANATOMY
       □ Policy loss: cross-entropy between network output and MCTS visit distribution
         L_π = -Σ π(a|s) × log(p(a|s))
         □ Is the MCTS distribution used as soft targets (all moves) or hard target
           (only the selected move)? Soft is correct AlphaZero approach.
         □ Is the policy output a raw logit or a probability?
           CRITICAL: CrossEntropyLoss in PyTorch expects RAW LOGITS, not probabilities.
           If you pass probabilities through CrossEntropyLoss it applies log twice.
           If you pass probabilities through NLLLoss without prior log, values are wrong.
           Trace exactly what is passed to exactly which loss function.
         □ Are illegal moves excluded from the loss calculation?
           (Targets for illegal moves should be 0, but gradient still flows through
           the logits — this might be intentional or a bug depending on implementation)
       
       □ Value loss: MSE or cross-entropy between network value and game outcome z
         L_v = (z - v)^2     [MSE approach]
         □ Is z always in the same range as v? (both ±1 for tanh, both [0,1] for WDL)
         □ If WDL (win/draw/loss as 3 classes): is the game result correctly encoded
           as a 3-class distribution? (draw = [0, 1, 0])
       
       □ Combined loss: L = L_v + c × L_π + L2_reg
         □ What is c (relative weighting)? AlphaZero uses c=1.0 (equal weight).
           Is this correct here?
         □ Is L2 regularization added? What is the regularization coefficient?
           AlphaZero uses λ = 1e-4. Is it applied to all weights? (Should exclude bias
           and BatchNorm parameters)

□ 5.2  OPTIMIZER CONFIGURATION
       □ Which optimizer? Adam? SGD with momentum? AdamW?
         AlphaZero original uses SGD with momentum 0.9 and weight decay 1e-4.
         Adam is also common in open source implementations.
       □ Current learning rate — what is it?
       □ Is there a learning rate schedule?
         □ Warmup phase?
         □ Decay type: step decay, cosine annealing, polynomial?
         □ When does decay trigger? After N steps? After ELO plateau?
       □ Gradient clipping: max_norm? Is it applied? To all params or just some?
         (Helps with training stability, especially early in training)
       □ Weight decay: is it applied? Correctly excluded from bias/BN?
         (AdamW implements weight decay correctly; vanilla Adam does not — 
          verify if using Adam + manual weight decay)
       □ Momentum parameters (β₁, β₂ for Adam): are they tuned or default?

□ 5.3  TRAINING STEP TRACE
       Trace one complete training step in detail:
       □ Data sampling from replay buffer — correct dtype? Correct device?
       □ Forward pass through model — output shapes correct?
       □ Loss computation — are inputs in correct form for each loss function?
       □ optimizer.zero_grad() called before backward? (If not: gradients accumulate
         across steps — this is almost certainly a bug)
       □ Loss.backward() — does this run without error?
       □ Gradient clipping — applied after backward, before optimizer step?
       □ optimizer.step() — any anomalies?
       □ Learning rate scheduler step — is it called? Is it called at the right time?
         (Some schedulers step per epoch, some per batch — wrong frequency = wrong LR)
       □ Check: is model in train() mode during training? eval() mode during inference?
         (BatchNorm behaves VERY differently in these modes)

□ 5.4  NUMERICAL STABILITY
       □ Check for potential log(0): any place where log is applied without a clamp?
         (log(0) = -inf → NaN in loss → all gradients become NaN → training dies)
       □ Check for potential 0/0: Q = W/N with N=0?
       □ Check for gradient explosion: are the initial losses reasonable?
         Policy loss should start near log(avg_legal_moves) ≈ log(30) ≈ 3.4
         Value loss (MSE) should start near 0.5 for random ±1 outputs
       □ Check loss curve if training has already been done: is it monotonically
         decreasing? Plateauing? Exploding? Collapsing to near-zero?
       □ Check if BatchNorm layers have any issues:
         - running_mean and running_var are updated during training
         - If eval() mode is used during training (bug), BN won't update stats

□ 5.5  MIXED PRECISION TRAINING
       □ Is torch.cuda.amp (Automatic Mixed Precision) used?
       □ If yes: is GradScaler correctly implemented?
         (Scale → backward → unscale → clip → step → update scale)
       □ If no: is it feasible to add? (RTX 5070 has excellent FP16/BF16 performance)
         Estimate speedup from adding AMP: typically 1.5-2.5× for conv-heavy models
       □ Are there any operations that don't support float16?
         (softmax, log_softmax, cross_entropy should run in float32 even with AMP)

□ 5.6  CHECKPOINT SYSTEM
       □ What is saved in a checkpoint? (model weights only? optimizer state? 
         scheduler state? iteration number? ELO? replay buffer?)
       □ Is optimizer state saved? (Without it, Adam's momentum is lost on resume,
         which hurts training for the first few hundred steps)
       □ Can training be resumed exactly from a checkpoint? Test this.
       □ Checkpoint naming convention — are old checkpoints overwritten or kept?
       □ Is the best model checkpoint tracked separately from the latest checkpoint?
       □ How often are checkpoints saved? Is it frequent enough to not lose work?
       □ Check: does checkpoint loading handle version mismatches?
         (Model architecture change between saves → checkpoint incompatible)

□ 5.7  TRAINING STABILITY MONITORING
       □ Is there logging of loss values during training?
       □ Is there detection of NaN/Inf losses? What happens when detected?
       □ Is there logging of gradient norms? (Useful for detecting explosion/vanishing)
       □ Is there evaluation (ELO estimation) at regular intervals?
       □ Are learning curves saved? (Or just printed to console and lost)

SELF-EXPAND: What training bugs or instabilities might have occurred in past training
runs based on the code you've read? Add investigation tasks.

═══════════════════════════════════════════════════════════
PHASE 6 — GPU & HARDWARE PERFORMANCE OPTIMIZATION
═══════════════════════════════════════════════════════════

□ 6.1  GPU UTILIZATION PROFILING
       □ Measure actual GPU utilization during:
         (a) Self-play (single game, 800 sims/move)
         (b) Self-play (if parallelized)
         (c) Batch training
         Use nvidia-smi, nvitop, or torch.profiler for measurement.
       □ Target: >80% GPU utilization during training.
         If <50%: CPU is the bottleneck — investigate where.
       □ GPU memory utilization: what fraction of 12GB is used during training?

□ 6.2  INFERENCE BATCHING
       □ During MCTS, are neural network evaluations batched?
         Single-position inference is massively GPU-inefficient.
         At batch size 1, the GPU is maybe 1-5% utilized.
         At batch size 256+, utilization can reach 80-95%.
       □ If not batched: this is likely the single biggest performance improvement
         available. Document how batching could be implemented (e.g., gather N leaf
         nodes before evaluating, use virtual loss to hold other threads waiting).
       □ If batched: what is the batch size? Is it tuned?
       □ Measure: inference latency at batch sizes 1, 8, 32, 128, 256, 512.
         Plot the throughput (positions/sec) vs batch size curve.
       □ Find the optimal batch size for RTX 5070 given this model architecture.

□ 6.3  DATA LOADING PERFORMANCE
       □ DataLoader configuration:
         - num_workers: how many? For GPU training, typically 4-8.
           Too few: data loading is the bottleneck.
           Too many: overhead exceeds benefit.
         - pin_memory: True or False? (True is faster for GPU transfer)
         - prefetch_factor: set? Default is 2.
         - persistent_workers: set? (Avoids worker restart overhead)
       □ Measure: data loading throughput (samples/sec)
       □ Is data loading the bottleneck? (Profile: are GPU cores waiting for data?)
       □ If replay buffer is stored as numpy arrays: are they properly vectorized?
         (No Python loops for sampling — use np.random.choice with replace=False)

□ 6.4  MEMORY TRANSFER OPTIMIZATION
       □ Are tensors pre-loaded to GPU or transferred per-batch?
       □ Is non_blocking=True used for tensor.to(device)?
       □ Are any tensors unnecessarily moved CPU↔GPU multiple times per step?
       □ Is there any device mismatch that causes silent CPU fallback?
         (e.g., one operation happening on CPU when everything else is on GPU)

□ 6.5  torch.compile() POTENTIAL
       □ Is torch.compile() applied to the model?
       □ PyTorch 2.0+ — torch.compile() can give 1.5-3× speedup on modern GPUs.
       □ Does the model structure support compilation? (Dynamic shapes can be
         problematic — MCTS with variable batch sizes may need special handling)
       □ If not applied: estimate potential speedup and recommend with mode options
         (mode="reduce-overhead" for small batch inference,
          mode="max-autotune" for training)

□ 6.6  CUDA GRAPHS
       □ Is the training loop a fixed computation graph?
         (Fixed batch size, fixed model size → CUDA graph capture is feasible)
       □ CUDA graphs eliminate CPU launch overhead for each CUDA kernel.
       □ For training steps: could give 10-30% speedup if CPU overhead is measurable.
       □ For inference at fixed batch size: could give 20-50% speedup.

□ 6.7  FP8 / BF16 POTENTIAL (Blackwell Architecture)
       □ RTX 5070 (Blackwell) supports FP8 inference.
       □ Is FP8 quantization feasible for inference in MCTS?
         (FP8 inference is ~2-4× faster than FP32, with minimal accuracy loss for
          inference-only scenarios)
       □ Is BF16 training being used? (BF16 has better dynamic range than FP16 for
         training — recommended over FP16 on Blackwell)

□ 6.8  PROFILING DEEP DIVE
       □ Run torch.profiler on one full training step.
         Identify the top 5 most time-consuming operations.
       □ Run torch.profiler on one MCTS search of 800 simulations.
         Identify the top 5 most time-consuming operations.
       □ For each of the top 5: is there an obvious optimization?

□ 6.9  SYSTEM RESOURCE USAGE
       □ CPU usage during training — is any core maxed out?
       □ RAM usage — is the replay buffer near RAM limits?
       □ Disk I/O — is checkpoint saving blocking training? Is replay buffer saved?
       □ Is there any resource leak? (RAM growing over time, GPU memory not freed)

SELF-EXPAND: Every bottleneck found → specific optimization task added to queue.

═══════════════════════════════════════════════════════════
PHASE 7 — EVALUATION & ELO MEASUREMENT SYSTEM
═══════════════════════════════════════════════════════════

□ 7.1  EVALUATION METHODOLOGY
       □ How is the model's strength measured during training?
       □ Is there a system for playing games against previous checkpoints?
       □ Is Bayesian ELO calculated? How?
       □ Is there evaluation against Stockfish at fixed skill levels?
         (Stockfish skill levels 0-20 with moveTime limits are standard benchmarks)
       □ Are evaluation games played without Dirichlet noise?
         (Noise would make evaluation non-deterministic and artificially weak)
       □ Are evaluation games played with greedy (τ=0) move selection?

□ 7.2  ELO ESTIMATION FRAMEWORK
       □ Document the current ELO estimation method if any.
       □ If none: recommend implementing a simple Elo tracking system:
         - Play 100 evaluation games vs Stockfish at calibrated skill levels
         - Use Bayesian Elo (BayesElo) or SPRT (Sequential Probability Ratio Test)
           for statistically rigorous comparison
       □ Check: is there hysteresis in model acceptance?
         (New model should be significantly better before replacing current best)
       □ Estimate: at current training throughput, how many steps until the model
         beats Stockfish at skill level 1? Level 5? Level 10?

□ 7.3  EVALUATION GAME QUALITY
       □ Are evaluation games logged/saved for analysis?
       □ Is there a way to visualize games? (PGN export? Board visualization?)
       □ Check early games: do they make obviously illegal moves? (Tests move legality)
       □ Check middle-training games: is there evidence of tactical awareness?
       □ Check late-training games (if available): are positions competitive?

═══════════════════════════════════════════════════════════
PHASE 8 — CODE QUALITY, MAINTAINABILITY & CORRECTNESS
═══════════════════════════════════════════════════════════

□ 8.1  DEAD CODE & TECHNICAL DEBT
       □ Find and document all TODO/FIXME/HACK/BUG/XXX comments with their context
       □ Find all commented-out code blocks — are these intentional or forgotten?
       □ Find all functions/classes that are defined but never called
       □ Find all imports that are never used
       □ Find any duplicate logic — same computation in multiple places

□ 8.2  MAGIC NUMBERS & HARDCODED VALUES
       □ Find every numeric constant in the code.
       □ Is each one named? Is the name self-documenting?
       □ Critical ones to flag: board dimensions, move encoding sizes, MCTS constants,
         training hyperparameters, device strings, file paths
       □ Are hyperparameters in a central config? Or scattered through files?

□ 8.3  ERROR HANDLING & EDGE CASES
       □ What happens if a checkpoint file is missing on startup?
       □ What happens if CUDA is not available? (Falls back to CPU? Crashes?)
       □ What happens if the replay buffer is empty and training is called?
       □ What happens if a game produces zero training positions?
         (Impossible but worth checking the guard)
       □ What happens on CUDA OOM? (Out of memory — is it caught gracefully?)
       □ What happens if the chess library raises an exception for an illegal move?
         (This would indicate a bug in move generation — should be caught and logged)

□ 8.4  REPRODUCIBILITY
       □ Are all random seeds set explicitly?
         - torch.manual_seed()
         - numpy.random.seed()
         - python random.seed()
         - torch.cuda.manual_seed_all()
       □ Is deterministic mode enabled? (torch.backends.cudnn.deterministic = True)
         Note: this slows training but enables exact reproducibility
       □ Is the code deterministic given the same seed? (Can you reproduce a bug?)

□ 8.5  LOGGING & OBSERVABILITY
       □ Is there a structured logging system (Python logging, wandb, tensorboard)?
       □ What metrics are logged? (loss, gradient norms, ELO, games/hour, steps/sec)
       □ Is logging to file? Only to console? (Console-only = logs are lost)
       □ Is there enough logging to diagnose a training failure after the fact?
       □ Is logging throttled appropriately? (Not so much that it slows training)

□ 8.6  TEST COVERAGE
       □ Are there unit tests? What is covered?
       □ Critical untested areas:
         - Move generation correctness
         - Board encoding/decoding roundtrip
         - MCTS backup correctness (manual trace test)
         - Value perspective consistency
         - Loss function inputs/outputs
       □ Are there integration tests? (Play a game to completion, check result is valid)
       □ Estimate: what percentage of known bugs would be caught by existing tests?

□ 8.7  PYTHON CODE QUALITY
       □ Is type hinting used? Consistently?
       □ Are docstrings present for key functions and classes?
       □ Run a linter (flake8/ruff) — how many warnings?
       □ Are there any circular imports?
       □ Are there any global mutable state variables outside of classes?
       □ Are there any bare except clauses that swallow errors silently?

SELF-EXPAND: 5 additional code quality tasks based on specific issues found.

═══════════════════════════════════════════════════════════
PHASE 9 — COMPARATIVE ANALYSIS WITH REFERENCE IMPLEMENTATIONS
═══════════════════════════════════════════════════════════

□ 9.1  ALPHAZERO PAPER COMPLIANCE
       Compare against the original AlphaZero paper (Silver et al., 2018):
       □ Input representation: paper uses 119 channels (8 history positions × 14 piece
         planes + 7 auxiliary). How many channels does this implementation use?
         Is the difference justified?
       □ Network size: paper uses 20 residual blocks × 256 channels.
         What does this implementation use? Is it scaled appropriately for hardware?
       □ MCTS simulations: paper uses 800 per move. What is used here?
       □ Batch size: paper uses 4096. What is used here?
       □ Replay buffer: paper uses 500,000 most recent positions. What is used here?
       □ Training window: paper starts training after 1000 positions in buffer.
         What is used here?
       □ Learning rate schedule: paper uses [100k: 0.2, 300k: 0.02, 500k: 0.002].
         What is used here?

□ 9.2  OPEN SOURCE COMPARISON
       Compare against known open source implementations:
       □ Leela Chess Zero (lc0): What architectural innovations does Lc0 use that
         could be applied? (WDL value head, policy with improved move encoding,
         attention mechanisms in newer networks)
       □ KataGo: What training improvements does KataGo use?
         (Auxiliary outputs, self-play curriculum, stronger draw handling)
       □ Simple AlphaZero: Compare against small educational implementations for
         correctness reference (e.g., suragnair/alpha-zero-general)
       □ Document: top 3 things this implementation does better than references.
         Document: top 3 things this implementation should adopt from references.

□ 9.3  NOVEL ELEMENTS
       □ Identify any approaches in the code that deviate from standard implementations
       □ For each deviation: is it a deliberate experiment? An accident? An improvement?
       □ Evaluate each: should it be kept, fixed, or reverted?

═══════════════════════════════════════════════════════════
PHASE 10 — STRESS TESTING & EDGE CASE PROBING
═══════════════════════════════════════════════════════════

□ 10.1  KNOWN TRICKY CHESS POSITIONS
        Run the following positions through the model and MCTS, verify behavior:

        □ Starting position (20 legal moves — verify count)
        □ Position with en passant available (verify it's in legal moves AND encoded)
        □ Position in check with limited escape squares
        □ Stalemate position (0 legal moves, not in check — verify draw detection)
        □ Checkmate position (0 legal moves, in check — verify immediate terminal return)
        □ Position where promotion is the only winning move
        □ Underpromotion position (knight promotion wins, queen promotion stalemates)
        □ Zugzwang position (any move worsens position — verify MCTS can handle this)
        □ King and pawns endgame (late game, few pieces)
        □ Position with 49 half-moves on clock (next move could trigger 50-move draw)

□ 10.2  MCTS STRESS TEST
        □ Run MCTS with 1, 10, 100, 1000, 10000 simulations — does behavior scale?
        □ Run MCTS on the same position twice with the same seed — identical output?
        □ Run MCTS on a terminal position — does it return immediately?
        □ Run MCTS on a position with only one legal move — does it choose it instantly?

□ 10.3  TRAINING STABILITY STRESS TEST
        □ Train for 1000 steps from scratch — does loss decrease?
        □ Check loss at step 0, 100, 500, 1000 — what are the values?
        □ Are there any NaN/Inf losses?
        □ Does the value head converge? (Should move from near 0 toward actual ±1)
        □ Does the policy head show any preference after 1000 steps?
          (At minimum, should prefer e4/d4/e5/d5 from the starting position)

□ 10.4  RECOVERY & RESUME TEST
        □ Start training, save checkpoint at step 100
        □ Resume from checkpoint, compare loss at step 101 to original run
        □ Are they identical? (Should be, given same random seed)
        □ If not: what state is not being properly saved/restored?

SELF-EXPAND: Every test failure → root cause investigation task added.

═══════════════════════════════════════════════════════════
PHASE 11 — SECURITY, SAFETY & OPERATIONAL CONCERNS
═══════════════════════════════════════════════════════════

□ 11.1  RESOURCE MANAGEMENT
        □ Are CUDA contexts properly managed?
        □ Are model references cleaned up after evaluation games?
        □ Is there any GPU memory fragmentation after long training runs?
        □ Is training robust to SIGINT (Ctrl+C)? Does it save a checkpoint?

□ 11.2  CHECKPOINT INTEGRITY
        □ Is there checkpoint validation on load? (Corrupt checkpoint → crash or
          silent bad weights?)
        □ Are checksums or hashes stored with checkpoints?
        □ Is there a "last known good" checkpoint separate from "latest"?

□ 11.3  LONG-RUN STABILITY
        □ Are there any counters that could overflow after many training steps?
          (32-bit integers wrap at ~2.1 billion — could become an issue over time)
        □ Does the replay buffer grow unboundedly if there's a bug?
        □ Does the tree grow unboundedly if tree reuse is implemented?
        □ Any file handles left open?

═══════════════════════════════════════════════════════════
PHASE 12 — TASK QUEUE EXHAUSTION & FINAL SWEEP
═══════════════════════════════════════════════════════════

By now, your task queue should have 50-100+ items. Many should already be completed.
Work through every remaining item.

For each item:
  □ Investigate fully
  □ Document the finding
  □ Assign it to a priority tier (CRITICAL / HIGH / MEDIUM / LOW)
  □ Generate the specific fix or recommendation

Do not declare this phase complete until the queue has fewer than 5 unresolved items,
and those 5 must be explicitly justified as "out of scope" or "requires hardware
not available."

═══════════════════════════════════════════════════════════
FINAL DELIVERABLE — THE COMPLETE AUDIT REPORT
═══════════════════════════════════════════════════════════

Produce the following structured report:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION A — EXECUTIVE SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

In 10 sentences or fewer: what is the overall state of this project?
What are the 3 most critical findings? What is the estimated improvement potential?

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION B — PERFORMANCE BASELINE (MEASURED NUMBERS)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

All numbers measured on RTX 5070. No estimates where measurements are possible.

  Neural Network:
    Parameters:                    ______
    Model size (MB):               ______
    Inference latency (batch=1):   ______ ms
    Inference latency (batch=256): ______ ms
    Throughput (positions/sec):    ______
    VRAM usage during training:    ______ GB / 12 GB
    Training step time:            ______ ms
    Training steps/hour:           ______

  MCTS:
    Simulations/second:            ______
    Time per move (800 sims):      ______ ms
    Bottleneck:                    ______

  Self-Play Pipeline:
    Average game length:           ______ moves
    Average game time:             ______ seconds
    Games/hour (current):          ______
    Games/hour (theoretical max):  ______
    Positions/hour:                ______

  Estimated Training Timeline:
    Time to 1M positions:          ______
    Time to beat Stockfish lvl 1:  ______ (estimated)
    Time to beat Stockfish lvl 5:  ______ (estimated)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION C — CRITICAL BUGS (Fix Immediately — Before Any Training)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For each: Description | Code Location | Impact | Exact Fix

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION D — HIGH PRIORITY (Fix This Week)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For each: Description | Estimated Impact | Implementation Effort | Recommendation

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION E — MEDIUM PRIORITY (Next Iteration)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION F — LOW PRIORITY & EXPERIMENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION G — ARCHITECTURE RECOMMENDATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

If fundamental architectural limitations are found: propose the cleanest path to
a better architecture, with concrete changes, estimated performance improvement,
and implementation order.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION H — PRIORITIZED IMPLEMENTATION ROADMAP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

A step-by-step plan, ordered by impact/effort ratio:

  Day 1:   [Critical bug fixes — list them]
  Day 2-3: [High priority performance fixes]
  Week 2:  [Architecture improvements]
  Month 2: [Long-term optimizations and experiments]

Expected performance after each stage:
  After Day 1:   ______ positions/hour (____×  improvement)
  After Week 1:  ______ positions/hour (____×  improvement)
  After Month 1: ______ ELO estimate

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION I — COMPLETE TASK QUEUE LOG
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Full list of every task that was created during this audit, with status and finding.
This serves as a complete audit trail.

═══════════════════════════════════════════════════════════
NON-NEGOTIABLE RULES — ALWAYS IN EFFECT
═══════════════════════════════════════════════════════════

1.  Never say "looks fine" without quoting the specific code and explaining WHY.
2.  Never say "performance is acceptable" without a measured number.
3.  Never say "this is probably not a bug" — investigate or explicitly defer.
4.  Always show your live task queue at the end of each phase.
5.  Always generate at least 3 new tasks at the end of each phase from what you learned.
6.  When you find a bug, trace EVERY system that it could affect.
7.  When you find a performance issue, calculate the exact impact on positions/hour.
8.  Never finish a phase just because it feels done — re-read what you wrote and
    ask: "What question did I not answer here?"
9.  If you are unsure whether something is a bug: treat it as a bug until proven otherwise.
10. The goal is to make this chess AI train faster, play stronger, and behave correctly.
    Every finding must map to one of these three outcomes.

BEGIN WITH PHASE 0.
Show the initial task queue (all phases checked as [ ] pending).
Update it after every sub-phase.
Do not stop until Section I is complete.