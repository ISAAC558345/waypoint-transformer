# Current multi-UAV spectrum audit (pre-Round-19)

This is a source audit of the locked Round-18 implementation, not a physical
assumption.  `src/wt/round18/multi_env.py::rate_matrix` calls
`compute_channel_metrics` separately for every UAV position, each time passing
the unchanged `MaritimeEnvConfig.total_bandwidth_hz`.

`src/wt/maritime/channel.py::compute_channel_metrics` computes, for each
UAV--buoy pair, `beta = reference_channel_gain / d_3d^pathloss_exponent`,
`SNR = buoy_tx_power_w * beta / noise_power_w`, and
`R[m,k] = (total_bandwidth_hz / K) * log2(1 + SNR)`.  The locked defaults are
`B=1.0e6 Hz`, `P_b=0.1 W`, `N=1.0e-9 W`, reference gain `1.0e-3`, and
path-loss exponent `2.2`.

There is no cross-UAV or interference helper in Round-18.  The decoder scores
`W[m,k]=(Q[k]/queue_capacity_bits)*R[m,k]` and chooses an injective assignment;
the queue capacity then uses the selected `R[m,k]*delta_t`.  Thus every UAV
gets its own independently computed full `B/K` rate vector.  For M=2, the
same configured total bandwidth is effectively reused at both receivers; two
different simultaneous uplinks have neither a shared-bandwidth constraint nor
cross-signal penalty.  This is why the communication portion of Round-18 is
approximately separable once the shared queue, exclusive assignment, and
safety projection are fixed.

Round-19 preserves this exact behavior as `spectrum_mode="orthogonal"`.
