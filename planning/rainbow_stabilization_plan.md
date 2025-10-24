# Rainbow DQN Stabilization Plan (complete)

This document enumerates targeted adjustments needed to satisfy the thirteen
stabilization requirements for the Rainbow DQN Blackjack agent. Each section
lists (a) the concrete issue in the current implementation with line-level
references, and (b) the precise code modifications (in pseudocode/description)
required to resolve it.

## 1. Fortify `NoisyLinear` initialization & reset semantics
**Observations**
- `reset_parameters` relies on `.data` mutations, which bypass AMP/autograd
  bookkeeping and can desynchronize parameter dtype/device.【F:agents/dqn_rainbow.py†L43-L81】
- `reset_noise` writes directly into buffers without `torch.no_grad()`, so it can
  be captured by gradient history or autocast scopes.【F:agents/dqn_rainbow.py†L68-L81】

**Required adjustments**
- Rewrite `reset_parameters` to wrap tensor fills in `with torch.no_grad()` and
  use in-place ops on the parameter tensors themselves (no `.data`).
- Extend `reset_noise` to guard the sampling with `torch.no_grad()`, and add an
  optional `disable: bool = False` flag so callers can zero the sampled noise
  during evaluation (see §2/§13). Sketch:
  ```python
  def reset_noise(self, disable: bool = False) -> None:
      with torch.no_grad():
          if disable:
              self.weight_eps.zero_(); self.bias_eps.zero_(); return
          eps_in = _scaled_noise(...); eps_out = _scaled_noise(...)
          self.weight_eps.copy_(torch.outer(eps_out, eps_in))
          self.bias_eps.copy_(eps_out)
  ```

## 2. Deterministic evaluation path for NoisyNets
**Observations**
- `_maybe_reset_noise()` unconditionally resamples noise and is invoked inside
  both `act_bet` and `act_play` even under `torch.no_grad()`, so evaluation never
  becomes deterministic.【F:agents/dqn_rainbow.py†L374-L403】

**Required adjustments**
- Introduce a context helper (e.g., `with disable_noisy(self.online_net): ...`)
  that temporarily calls `reset_noise(disable=True)` on all `NoisyLinear`
  modules and restores training mode afterwards.
- Update `act_bet`/`act_play` to skip resampling when invoked from evaluation
  (pass a flag or detect `self.online_net.training is False`). Ensure epsilon
  stays at 0 (already handled by `epsilon()` returning 0 when `use_noisy`).

## 3. Refresh noise before every forward during training
**Observations**
- `_maybe_reset_noise()` is called only once before the first forward in
  `_loss_from_batch`, so the online network reuses the same noise when it is
  invoked again for next-state evaluation and when the target network is
  queried.【F:agents/dqn_rainbow.py†L495-L519】

**Required adjustments**
- Inside `_loss_from_batch`, call `reset_noise_all(self.online_net)` (new helper
  that loops `modules()` and calls `reset_noise()`) before **each** forward of
  the online net, and likewise refresh the target net before its forward.
- Ensure the helper respects training/eval mode and runs under `torch.no_grad()`
  when resetting the target network.

## 4. Preserve probability mass when `l_idx == u_idx`
**Observations**
- The C51 projection scatters mass to lower/upper bins only; when `b` is an
  integer, both `(u - b)` and `(b - l)` are zero and the distribution `m`
  becomes all zeros, which later forces renormalization clamps and introduces
  NaNs/1e9 spikes.【F:agents/dqn_rainbow.py†L520-L530】

**Required adjustments**
- After computing `l_idx`/`u_idx`, branch on `same = (u_idx == l_idx)` and
  scatter the entire `target_probs` mass into that index before handling the
  fractional case.
- Keep the projection in fp32 (see §5) to avoid rounding mismatches that cause
  spurious `same` detections.

## 5. Execute distribution math in fp32 even with AMP
**Observations**
- `torch.softmax`/`log_softmax` run inside `autocast_if`, so logits stay in fp16
  on AMP runs, causing severe underflow with 51 atoms.【F:agents/dqn_rainbow.py†L501-L536】

**Required adjustments**
- Cast logits to fp32 before calling `log_softmax`/`softmax`, e.g.:
  ```python
  logits = play_outputs.float()
  log_probs = torch.log_softmax(logits, dim=-1)
  probs = torch.softmax(logits, dim=-1)
  ```
- Apply the same cast for `next_online_logits` and `next_target_logits` before
  computing expectations/projections. Maintain fp32 for `support`, `m`, and any
  reductions before exiting the autocast region.

## 6. Renormalize distributions after stability clamps
**Observations**
- `normalize_probs` clamps the post-softmax tensor to `[1e-6, 1.0]` but never
  re-normalizes, distorting expectations and the C51 cross-entropy target.
  【F:agents/dqn_rainbow.py†L96-L99】

**Required adjustments**
- Update `normalize_probs` to clamp **then** divide by the per-row sum so each
  distribution sums to 1. Keep the operation in fp32 to match §5.

## 7. Keep TD errors & PER priorities finite
**Observations**
- In the C51 branch, `q_sa` is gathered from `mask_q(q_expectation, legal)`;
  when `legal` is false, the gather returns `-1e9`, producing extreme TD errors
  and flat priority updates.【F:agents/dqn_rainbow.py†L501-L566】
- `update_priorities` writes raw TD errors without a floor, so zero/NaN entries
  can kill sampling probabilities.【F:agents/replay.py†L100-L153】

**Required adjustments**
- Mask out illegal actions before computing TD errors: compute a boolean
  `valid_sa` from the mask gather and replace invalid entries with zeros prior to
  `td_error` (and log a warning if any invalids appear).
- Apply a stability floor when updating PER priorities (`np.maximum(td_error,
  eps_prio)`) and clamp excessively large values (e.g., `np.minimum(..., 1e6)`)
  inside `PrioritizedReplayBuffer.update_priorities`.
- When PER is enabled, normalize importance weights by `w / (w.max() + 1e-8)` in
  `_loss_from_batch` to avoid underflow; the `sample` method already divides by
  `weights.max()` but re-verify the GPU path returns contiguous fp32 weights and
  clamp again after casting to torch.

## 8. Compute scalar loss outside autocast & in fp32
**Observations**
- `per_sample.mean()` is executed inside `autocast_if`, allowing AMP to downcast
  the reduction to fp16, which can zero-out the loss.【F:agents/dqn_rainbow.py†L555-L566】

**Required adjustments**
- Move the scalar mean outside the autocast block: inside autocast compute
  `per_sample` in fp32, exit the context, then call `loss = per_sample.mean()
  .to(torch.float32)`.
- Ensure any diagnostics derived from `per_sample`/`td_error` use fp32 copies
  (call `.float()` before logging) so AMP does not underflow the statistics.

## 9. Harden AMP scaling & grad handling
**Observations**
- `train_step` calls `scaler.scale(loss).backward()` but never checks for finite
  gradients before stepping; NaNs propagate directly into parameters. There is
  also no early-exit when the scaled loss is non-finite.【F:agents/dqn_rainbow.py†L568-L606】

**Required adjustments**
- After `backward()`, call `self.scaler.unscale_(self.optimizer)` **before**
  grad clipping, then compute a grad norm (used for telemetry §11).
- Insert a finite-check:
  ```python
  if not torch.isfinite(loss.detach()):
      self.optimizer.zero_grad(set_to_none=True)
      self.scaler.update()  # drop the invalid scale
      diagnostics["nan_skipped"] = torch.ones(1)
      return loss.detach(), per_sample.detach(), td_error.detach(), diagnostics
  ```
- Only call `self.scaler.step(self.optimizer)` when grads are finite, and never
  call `optimizer.step()` directly when AMP is enabled (already true; keep guard
  for the non-AMP path).

## 10. Enforce dtype/device consistency for replay batches
**Observations**
- `_to_device` forwards tensors without enforcing dtype; when the GPU replay
  stores fp16 states (because `use_amp=True`), they flow into the network in
  half precision, conflicting with the fp32 math in §5-8.【F:agents/dqn_rainbow.py†L101-L112】【F:agents/dqn_rainbow.py†L495-L519】
- In the GPU replay path, rewards/states adopt the buffer dtype (potentially
  fp16).【F:agents/replay.py†L31-L123】

**Required adjustments**
- After `_to_device`, explicitly cast `states`, `next_states`, and `rewards` to
  `torch.float32`, `dones` to `torch.float32`, and masks to `bool`. Ensure `legal`
  tensors are contiguous before masking.
- When `replay_on_gpu` is true, cast tensors to fp32 on read (`self.states[indices]
  .to(dtype=torch.float32)` etc.) before returning the batch dictionary.

## 11. Expand telemetry & add NaN watchdog
**Observations**
- Logged metrics omit required diagnostics: grad norm, non-zero grad count,
  learning rate, replay fill ratio, PER α/β, scaler state, and NaN warnings.
  【F:agents/dqn_rainbow.py†L614-L626】

**Required adjustments**
- After each update (and within the finite-check path), compute:
  - `grad_norm = torch.linalg.vector_norm(torch.stack([...]))`
  - `grad_nonzero = sum((p.grad != 0).sum()` for parameters)
  - Current LR from `self.optimizer.param_groups[0]["lr"]`
  - `per_alpha = self.buffer.alpha`; `per_beta = self.buffer.beta`
  - Flag `nan_detected`/`inf_detected` when checks fail
- Merge these values into the `metrics` dict emitted every `log_interval`. For
  heavy computations, reuse the grad norms from §9 to avoid duplicate loops.

## 12. Replay warmup & optional prefill policy
**Observations**
- Training simply waits for `buffer.pos >= min_buffer_size`, with no accelerated
  warmup or callback hook to inject a scripted policy.【F:agents/dqn_rainbow.py†L568-L715】

**Required adjustments**
- Add a configuration option (e.g., `prefill_policy: Literal["random", "basic", None]`)
  and `prefill_steps` to `AgentConfig`/trainer. Implement a helper
  `prefill_buffer(policy_name, steps)` that runs before the training loop.
- Implement at least a random policy using env masks; optionally stub a basic
  blackjack heuristic. Guard so the helper is skipped when `min_buffer_size` is 0
  or `prefill_policy is None`.
- Update warmup telemetry to report progress (ratio) during prefill.

## 13. Deterministic evaluation wrapper & API
**Observations**
- There is no dedicated evaluation method; consumers must call `act_bet`/
  `act_play` directly, which resample NoisyNet noise and may leave the model in
  training mode.【F:agents/dqn_rainbow.py†L374-L715】

**Required adjustments**
- Add an `evaluate(env, hands, callback=None)` method that:
  1. Switches `online_net` to `eval()` and disables Noisy noise via the context
     helper from §2.
  2. Runs greedy action selection (no epsilon) for the requested number of hands.
  3. Restores training mode and reenables noise afterwards.
- Ensure the evaluation path logs summary statistics (win rate, average reward,
  bankroll deltas) and respects deterministic execution for reproducibility.

---

### Verification checklist
After implementing the above changes, run the following experiments (per
acceptance criteria) to validate stability:
1. Short training dry-run (≈50k steps, `min_buffer_size=10k`) to confirm no NaNs
   and healthy `td_error_std`/`q_std` metrics; inspect new telemetry fields.
2. A/B comparison for 100k steps: `use_noisy=True` (ε=0) vs `use_noisy=False`
   (restore ε-greedy) to ensure NoisyNets learns at least as quickly.
