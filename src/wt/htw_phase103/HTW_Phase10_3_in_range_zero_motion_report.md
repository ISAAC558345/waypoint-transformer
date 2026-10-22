# HTW Phase-10.3 服务区内零位移门控闭环验证

未训练。未改环境/奖励/hold。design evaluation，冻结单模型。

1. 规则：动作前若非idle、目标有效、queue>0 且 env._in_range，则 movement 输入改为 (0,0)；否则沿用执行器输出。
2. 均值 payload B0/B1/B2/B3 = 8500000/7250000/12875000/10750000。
3. B2-B1=5625000；B3-B0=2250000；B2-B3=2125000。B2-B0=4375000。recovery=(B2-B1)/(B0-B1)=4.5（不对缺口截断；不能代替 B2 vs B3）。
4. 完成块均值 B0/B1/B2/B3=8.5/7.25/12.875/10.75。
5. 换目标丢失事件均值 B0/B1/B2/B3=0.625/4.375/0.125/0.625；丢失rho=0.266/0.875/0.032/0.195。
6. gate 比例 B2/B3=0.3759/0.3772；B0/B1=0/0。零输入的实际执行位移均为 0。
7. 仍有圈内停住未完成块的时隙（B2 `gated_in_range_no_block` 约 89–101/场景）；部分低速率门控未完成见 CSV。从未完成 UAV-episode 均值 B2=3.5，并不优于 B0=2.5。
8. 情况B：门控把 PPO 和 fly-toward 都抬上去了；B2 均值不低于 B3。可保留该冻结执行门控做后续独立验证。不是 PPO 重新训练后学会驻留。
9. 不能声称 PPO 优于规则：仅单模型、已用过的 design 8–15，无跨 seed / 无独立 confirmation。
10. 下一步：多 seed、未用于设计的独立场景复核；本轮不自动运行、不调阈值、不训练。

运行 32 个 episode（B0/B1 重放核验 wrapper 关闭后仍为 8.50e6 / 7.25e6），墙钟约 7.12s。
图：`src/wt/htw_phase103/outputs/gate/phase103_payload_distance.png`（标注 design evaluation，冻结单模型）。
