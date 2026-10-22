# HTW Phase-9.4.4 Working Rules

Diagnostic-only HGNN encoder / training instability audit.

- Freeze Phase-9.4.3 order-free decoder checkpoints. Do not modify HGNN, decoder,
  set likelihood, optimizer, learning rate, batch size, epochs, L3-B, SINR, or
  Task-Waypoint.
- No PPO. No model selection on Phase-9.4.3 confirmation payloads.
- Phase-9.4.3 confirmation is consumed for post-hoc mechanism diagnosis only.
  Later model changes must use a new independent confirmation.
- Do not drop seeds 3/4 or report only seeds 0/1/2.
- Official 943 weights stay read-only. This package only writes under
  `src/wt/htw_phase944/outputs/` and the Phase-9.4.4 report.
