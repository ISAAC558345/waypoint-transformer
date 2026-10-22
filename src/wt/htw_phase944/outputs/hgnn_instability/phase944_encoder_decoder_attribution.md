# Encoder vs decoder attribution

未跨 seed 交换 decoder（embedding basis 不兼容）。比较进入 decoder 之前的 embedding / first-step logits。

- deepsets seed=0: uav_norm=18.688 logit_var=261.857 margin=8.572 idle_margin=-4.694 partial_ctx uses set decoder contexts
- deepsets seed=1: uav_norm=16.415 logit_var=106.490 margin=8.260 idle_margin=-3.435 partial_ctx uses set decoder contexts
- deepsets seed=2: uav_norm=16.959 logit_var=507.885 margin=6.829 idle_margin=-3.456 partial_ctx uses set decoder contexts
- deepsets seed=3: uav_norm=17.240 logit_var=357.690 margin=8.634 idle_margin=-1.418 partial_ctx uses set decoder contexts
- deepsets seed=4: uav_norm=17.655 logit_var=366.947 margin=6.843 idle_margin=-2.935 partial_ctx uses set decoder contexts
- hgnn seed=0: uav_norm=20.465 logit_var=389.629 margin=5.224 idle_margin=-5.206 partial_ctx uses set decoder contexts
- hgnn seed=1: uav_norm=23.524 logit_var=184.788 margin=4.915 idle_margin=-5.687 partial_ctx uses set decoder contexts
- hgnn seed=2: uav_norm=25.115 logit_var=387.905 margin=7.196 idle_margin=-4.090 partial_ctx uses set decoder contexts
- hgnn seed=3: uav_norm=21.821 logit_var=395.588 margin=6.392 idle_margin=-2.769 partial_ctx uses set decoder contexts
- hgnn seed=4: uav_norm=18.757 logit_var=202.932 margin=5.831 idle_margin=-2.890 partial_ctx uses set decoder contexts
