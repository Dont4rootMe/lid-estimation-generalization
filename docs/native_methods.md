# Native learning tasks (v9), point inputs (v11), scalar NF conditioning (v12)

Current experiment protocol v13 excludes PFGM++ by the author's decision:
39 representations ×11 active methods =429 cells. Its implementation, clipped
kernel and tests remain available. Matrix construction, launch admission,
group completeness and score-table generation use the active roster; disabled
PFGM++ is not counted as a failed/missing experimental result. The equations
below retain the implemented method for reference. No kernel/readout substitution
or additional training accompanies this exclusion.

Implementation: `models/native_tasks.py`, `models/native_bridge.py`,
`configs/fair_comparison/native_tasks.yaml`. Entry point:
`experiments.fair_protocol.resolve` followed by `models.training.train_model`.
This document describes implemented learning tasks. No learning experiment was
run during this repair, and convergence or LID quality is not certified.

The previous universal posterior wrapper made eight interfaces the same
finite-parameter objective. The new routes use the supplied native input and
condition, and a native raw network output. Conversion to a posterior-like
field happens only in `predict_lid`. Old checkpoints remain on their recorded
legacy routes; they are not native-v9 results. `direct_rectified_flow` is retired
from the active roster because it duplicates independent Gaussian RF.

## Definitions and architecture

Let X be a clean, commonly normalized observation in d supplied coordinates;
Z is an independent standard Gaussian vector. F denotes the raw neural-network
output. MSE below means coordinate mean followed by batch mean, unless a
coordinate sum is explicitly stated. No covariance projection or analytically
supplied normal-space prediction is used. One common scalar data normalization
is fitted on source-train data. From protocol v11, vector fields use
`PointResidualCore`: the first linear layer receives exactly the supplied native
point coordinates, without spatial sine/cosine features or their attenuation.
Width512, four residual blocks and separate native time/noise conditioning are
retained. Sinusoidal encoding of the scalar condition is not a spatial feature.
Images use the existing width-8 U-Net. These architectures are benchmark substitutions. An external
skip, scale, or coordinate transform is part of the mathematical task, not a
reason to call two tasks equivalent because their backbones have the same name.

The author explicitly rejected spatial Fourier features on September12. This
supersedes the earlier proposal to restore their noise filter. The shared30D
field now has2,698,014 parameters instead of3,005,214; DSB has two such cores.
NF capacity is recomputed against the actual common core under the existing10%
rule; it remains one sigma-conditioned invertible model. Learning targets,
native paths and loss weights are unchanged, as is the v10 model-specific
logarithmic scale-selection policy. The serialized `field_backbone` distinguishes
new point-input checkpoints from historical `spectral_residual_v1` checkpoints;
old weights retain their original inputs on load and are not converted results.
No learned-quality claim follows from this architecture change without a refit.

Protocol v12 additionally removes every sinusoidal/Fourier condition feature
from both vector and image NF. A learned MLP receives the single log(sigma).
The existing `fourier_features=0` architecture setting records this choice;
historical positive values reconstruct historical checkpoints. New native NF
training rejects positive values before data loading or optimization.
One sigma-conditioned NF, its NLL, Gaussian corruption and input scaling
by sqrt(1+sigma²) are retained. The latter scaling is an explicit additional
parameterization; frequency removal does not remove it. Capacity is recomputed
by the unchanged10% policy: vector30D NF width512/8 couplings has2,830,960
parameters; image NF uses two width5 conditioners with91,300 grayscale or
91,848 RGB parameters. Field backbones and their time conditioning are unchanged.

## Implemented rows

| Existing ID | Native input and condition | Raw target and loss | Status / source |
|---|---|---|---|
| `vp_diffusion` | Y=a(t)X+b(t)Z, condition t uniform in [0,1); a=exp[-(.1t+9.95t²)/2], b=sqrt(1-a²) | F predicts -Z; E sum_j(F_j+Z_j)² | FLIPD negative-noise regression, VP SDE [1,2] |
| `ve_diffusion` | Y=X+sigma(t)Z, sigma=.01·5000^t, t uniform in [0,1); condition t | F predicts the same -Z; feature-summed MSE | FLIPD learning objective with corrected forward corruption [1,2] |
| `rectified_flow` | Y=tX+(1-t)Z, t uniform in [.001,1); condition 999t | F predicts X-Z; unweighted MSE | Independent-pair, one-round Gaussian RF; no reflow [3] |
| `scale_conditioned_nf` | Y=X+sigma Z, loguniform sigma; one invertible model conditioned on sigma | Exact negative log likelihood of Y given sigma | Explicitly approved sigma-conditioned RealNVP adaptation [7] |
| `schrodinger_bridge` | Trajectories between independent standard Gaussian source and supplied data; native elapsed time | Two networks predict transition increments; alternating IPF regression detailed below | Brownian DSB [4]; not the old terminal-denoising shortcut |
| `posterior_rectified_flow` | Y=(X+lambda Z)/(1+lambda); condition log(lambda) | F predicts X; lambda^-2 MSE | Author control, not a separately published vanilla algorithm |
| `direct_log_noise_affine_flow` | Y=X+lambda Z; condition log(lambda) | F predicts lambda Z, the velocity for increasing log(lambda); lambda^-2 MSE | Author control |
| `posterior_log_noise_affine_flow` | Y=X+lambda Z; condition log(lambda) | F predicts X; lambda^-2 MSE | Author control |
| `direct_vp_trigonometric_flow` | Y=sin(pi t/2)X+cos(pi t/2)Z; t uniform in [0,1), condition t | F predicts (pi/2)[cos(pi t/2)X-sin(pi t/2)Z]; unweighted MSE | Trigonometric InterFlow's uniform-time theoretical objective [5] |
| `posterior_vp_trigonometric_flow` | Y=(X+lambda Z)/sqrt(1+lambda²); condition log(lambda) | F predicts X; lambda^-2 MSE | Author control; not vanilla direct InterFlow |
| `t_flowmatching` | Y=tX+(1-t)eta; uniform t in [0,1); condition 1-t; eta=Z/sqrt(U/nu), U~chi²(nu) once per vector, nu=5 | F predicts eta; unweighted MSE | Native t-Flow noise regression [6] |
| `pfgmpp` | Y=X+noise from the author's clipped radial kernel; log(sigma)~N(-1.2,1.2²) | Native EDM residual F and weighted denoising loss, detailed below | PFGM++ clipped implementation with augmented dimension 128 [8] |

The four author controls retain loguniform lambda in [1/256,64] and their
declared weights. They are accepted controls, not four additional published
algorithms. A common population posterior optimum does not imply identical
finite-network objectives. Conversely, mathematical duplicates must not be
made artificially different: this is why the extra direct linear RF row is
removed. The InterFlow choice is the uniform-time theoretical objective;
Appendix I's dataset-specific Beta time samplers are not claimed reproduced.

## FLIPD VE correction

At FLIPD source pin [1], `models/diffusions/sdes/sdes.py`,
`VeSde.solve_forward_sde` draws the returned eps independently
of the noise used in x_end and drops x_start from x_end. A negative-noise target
from that eps is not regression on the noise that corrupted the datum.
The author explicitly instructed us to implement the correct mathematics.
Here Y=X+sigma Z and target=-Z use the same Z. Source min/max sigma, uniform
time sampling and feature-summed loss are retained. This is an explicit bug
correction, not byte-for-byte reproduction of the defective function.

## Brownian DSB

The reference has diffusion variance 2, N=20 steps, dt=.01, and T=.2. Source
is an independent standard Gaussian; terminal law is the supplied data.
The existing bridge row now contains two full-sized fields. Full budget uses
20 complete IPF rounds (40 half-passes); a separately requested preflight uses
one full round. Budgets must divide into equal half-passes. Only one field
receives gradients in a half-pass, while its opposite supplies trajectories.

The mean_match=False regression in [4] is implemented as follows. For the
opposite frozen transition mean m(x,t)=x+F_opposite(x,t), draw
x_new=m(x,t)+sqrt(2dt)Z. Regress the active increment network at
(x_new,T-t-dt) on m(x,t)-m(x_new,t). The first projection uses m(x,t)=x,
the Brownian reference. A uniform transition index is sampled per trajectory.
Fresh trajectories replace the released implementation's finite cache; this
preserves the regression distribution and avoids hidden cache state on resume.
The released 2D recipe uses an OU initialization; the explicitly chosen
reference here is Brownian, matching our manuscript's bridge theorem.

Following use_prev_net=False, reset the active field and optimizer moments
at each new half-pass. Keep the final IPF pair; early validation losses from
different projection tasks cannot select a comparable best pair. No EMA crosses
IPF tasks. Forward drift is F_dataward/dt, not the raw increment.
Time discretization and finite IPF rounds are numerical approximations, not an
exactly solved continuous bridge.

The manuscript already defines the Brownian forward-drift readout for general
bridge factors, not only the old Gaussian-smoothing special case. With
tau=lambda²/2 and t=T-tau, q(x,lambda)=x+tau*b_plus(x,t).
Response is div_x q and Full is Response+||q-x||²/lambda². These are direct
network derivatives; no local regression of neighboring sample points is used.
The accepted derivative support is sqrt(2dt)<=lambda<=sqrt(2T). A continuous
time-conditioned network interpolates between trained discrete times.

## PFGM++ sampler, denoiser, and remaining readout gate

For d supplied coordinates and D=128, draw B~Beta(d/2,D/2) in float64,
set U=clip(B,.001,.999), and use radius
sigma*sqrt(D)*sqrt(U/(1-U+1e-8)+1e-8), with independent uniform direction.
Both epsilon terms and the location of clipping match [8]. Torch Gamma draws
generate the same Beta distribution with checkpointed RNG state. There is no
unannounced replacement with an unclipped Student kernel.

With s_data=.5, the denoiser is
D_theta(Y,sigma)=c_skip Y+c_out F(c_in Y,log(sigma)/4), where
c_skip=s_data²/(sigma²+s_data²), c_out=sigma*s_data/sqrt(sigma²+s_data²),
and c_in=1/sqrt(sigma²+s_data²). Loss is the coordinate-mean squared error
to X, weighted by (sigma²+s_data²)/(sigma*s_data)². The lognormal sampler is
not truncated to the evaluation grid. The LSUN-only sigma rescaling in the
source is not applied to unrelated benchmark datasets.

Clipping changes the actual second moment, especially at low ambient d.
The RMS multiplier is computed once by deterministic Beta quadrature,
including the two clipping atoms. Thus lambda labels actual coordinate RMS,
not the unclipped Student approximation sqrt(D/(D-2))*sigma.

**Unresolved:** the old exact Student Full identity is not established for
this clipped kernel. The broad endpoint theorem and that particular exact
identity are different claims. Training and Response are implemented, but
new PFGM++ Full requests raise a descriptive error. The Full-primary campaign
rejects PFGM++ before training, rather than waste a run or substitute Response.
This row is not a completed Full benchmark integration. Resolving it requires
the clipped-kernel derivation or an explicit measurement-protocol decision.

The exact mixed measure is now implemented in `models/pfgm_kernel.py`:
continuous spatial density, two spherical atoms, radial CDF, moment integration
and the weak scale derivative including shell motion. RMS calibration uses
this implementation. The author's corruption sampler and native loss remain
unchanged. `log_continuous_density` deliberately does not claim to be a full
ordinary PDF: clipping atoms cannot be encoded by a Lebesgue density.

A no-training counterexample now shows why the Student-style local closure
cannot simply be extended to this kernel. Two positive Gaussian-mixture data
laws have the same p, q, div(q), and d_log_sigma(q) at one query/scale, to1.2e-16,
but Full differs by0.00343707524. Their marginal scores differ. This excludes
an exact formula using only these local observables; it does not exclude every
possible global density-reconstruction algorithm. The general identity
Full=div(q)+(q-x)·score still holds when the convolved data density is smooth.
The current denoising head does not separately supply that marginal score.
Reproducible evidence: `diagnostics/pfgm_clipped_kernel_20260912/check_full_closure.py`
and its saved JSON. Run the script from this checkout; it performs deterministic
quadrature and linear algebra only, without fitting a neural network.
Full remains gated; supplying Response or the unclipped identity under its name
would change the requested comparison. No such substitution has been made.

## Evaluation and common benchmark choices

Gaussian routes reconstruct q from their native trained outputs only during
evaluation. At canonical query x and physical lambda, let a=1/sqrt(1+lambda²),
b=lambda*a. The adapters are:

- VP: q=x+lambda*F(a*x,t(lambda)); VE: q=x+lambda*F(x,t(lambda)).
- RF: t=1/(1+lambda), q=t*x+(1-t)*F(t*x,999t).
- Direct log-noise: q=x-F(x,log(lambda)); its posterior control uses F directly.
- Direct trig: t=2*atan(1/lambda)/pi, q=a²*x+(2b/pi)*F(a*x,t).
- Posterior linear/trig controls: q=F(a_path*x,log(lambda)).
- t-Flow: s=lambda/sqrt(nu/(nu-2)), t=1/(1+s), q=x-s*F(t*x,1-t).
  Full then uses the existing Student scale-derivative identity, not Gaussian Full.
- PFGM++: q is the native EDM denoiser at its calibrated sigma; Full is gated above.
- NF: same single conditional density, exact log-scale derivative and OLS5;
  no separate flow per sigma. The five-point scale fit is retained explicitly.

For Gaussian routes Full=div(q)+||q-x||²/lambda². All input transforms remain
inside the differentiated evaluation graph, so native Jacobian scale factors
are included. Vector trace is exact; image trace uses the declared Hutchinson
budget. This conversion does not enter training.

Protocol v10 uses 29 geometric candidates per method, including both bounds:
lambda[k] = L * (U/L)^(k/28), k=0,...,28. The finite native support defines L,U;
zero lower bounds use 1/256 and infinite upper bounds use the explicit search
cap 64. NF center bounds account for its five-point OLS5 stencil. These are
evaluation choices; no training support, objective or DSB horizon is changed.

| Method | Evaluation window (rounded for display) | Candidates |
|---|---|---:|
| VP / FLIPD | [1/256, 152.1669703] | 29 |
| VE / FLIPD | [0.01, 50] | 29 |
| Gaussian RF | [1/256, 999] | 29 |
| Conditional NF, OLS5 centers | [1/256, 64] | 29 |
| DSB | [sqrt(0.02), sqrt(0.4)] | 29 |
| All four author controls | [1/256, 64] | 29 |
| Direct trigonometric flow, t-Flow, PFGM++ | [1/256, 64] | 29 |

The [1/256,64] grid retains the original half-octaves exactly. Other windows
are regenerated, not obtained by filtering that grid. The primary supervised
and pointwise automatic selectors share the same 29 candidates within a method.
Unknown-LID reference selection also uses these 29 candidates, then freezes
and transfers the chosen lambda to dependent cells. Both primary Kneedle
selectors use log(lambda), retaining convex/decreasing, S=1, offline interp1d
and their existing fallback/failure rules. A denser grid is not evidence that
Kneedle is accurate; boundary choices remain reported. PFGM++ Full stays blocked.

Pre-v10 receipts retain their original grids and reference-VP-time coordinate
when replayed. Fresh receipts and exports explicitly identify the v10 policy.
Historical 22-time diagnostics are also masked and labeled as filtered when
necessary. Boundary selections/fallbacks remain reported, not quality passes.

Shared benchmark choices remain 128000 AdamW updates, batch 256, learning
rate .0002, weight decay 1e-6, clipping norm 1, final cosine decay, and matched
preprocessing/splits. These are common optimization-budget choices, not a
claim to reproduce every original paper's optimizer, architecture, or dataset
recipe. DSB has two cores and extra trajectory cost; equal FLOPs are not claimed.
Other fields keep the common EMA/best-holdout policy; DSB uses its final pair.

## Sources

Verification on September 12: 135 distinct focused checks passed across native
equation/gradient/serialization, fair comparison/measurement/export and backbone
capacity suites. The genuine optimizer-update test was excluded. Training-entry
fixtures replace optimizer.step with an exception, so no parameter update is
performed. These results verify implementation mechanics, not learned quality.

1. [FLIPD source, pin 05ab170](https://github.com/layer6ai-labs/flipd/tree/05ab170c2c9bcada3f9286c3ef86db31b80925fa): shared Lightning loss, VP/VE SDEs.
2. [Song et al., Score-Based Generative Modeling through SDEs](https://arxiv.org/abs/2011.13456): VP/VE processes and score matching.
3. [Liu et al., Flow Straight and Fast](https://arxiv.org/abs/2209.03003), [RF source pin 5a1fd4d](https://github.com/gnobitab/RectifiedFlow/tree/5a1fd4dd3ea7db764ce370a84ce35f9c8b15fde6).
4. [De Bortoli et al., Diffusion Schrödinger Bridge](https://arxiv.org/abs/2106.01357), [DSB source pin 1c82eba](https://github.com/JTT94/diffusion_schrodinger_bridge/tree/1c82eba0a16aea3333ac738dde376b12a3f97f21): `bridge/langevin.py`, `bridge/runners/ipf.py`, `conf/dataset/2d.yaml`.
5. [Albergo and Vanden-Eijnden, Building Normalizing Flows with Stochastic Interpolants](https://arxiv.org/abs/2209.15571): trigonometric interpolant and velocity objective.
6. [t-Flow, 2410.14171](https://arxiv.org/abs/2410.14171): Appendix B noise regression and native conditioning.
7. [Dinh et al., Density Estimation Using Real NVP](https://arxiv.org/abs/1605.08803): exact invertible density objective; sigma conditioning is our approved adaptation.
8. [Xu et al., PFGM++](https://arxiv.org/abs/2302.04265), [source pin d57c1ee](https://github.com/Newbeeer/pfgmpp/tree/d57c1ee4488d8e4064a5b9c8548792f00395aa8b): `training/loss.py` and EDM preconditioning.
