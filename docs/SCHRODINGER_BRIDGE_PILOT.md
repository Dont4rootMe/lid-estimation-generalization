# Canonical Brownian Schrödinger-bridge pilot

The trainable bridge job uses one fully specified generalized Brownian bridge,
rather than an unspecified or independent endpoint coupling. Its terminal law
is the dataset distribution \(\mu_T\), its Schrödinger factors are
\(F=\mathrm{Lebesgue}\) and \(G=\mu_T\), and its initial law is therefore

\[
\mu_0 = h_{\gamma T} * \mu_T.
\]

At time-to-go \(\tau=T-t\), the bridge marginal is
\(h_{\gamma\tau}*\mu_T\). This factorization defines a probability coupling
even when the terminal data law is singular; it is the generalized mixture of
Brownian bridges covered by the paper.

The network is conditioned directly on `tau`. Training samples

\[
X_T\sim\mu_T,\qquad
R_\tau=X_T+\sqrt{\gamma\tau}\,Z,
\]

and learns the terminal denoiser \(\widehat X_T(R_\tau,\tau)\). The exported
forward drift and its divergence are

\[
\widehat b^+(r,\tau)=\frac{\widehat X_T(r,\tau)-r}{\tau},
\qquad
\operatorname{div}\widehat b^+
=\frac{\operatorname{div}\widehat X_T-D}{\tau}.
\]

The primary full-density readout is

\[
\widehat d_{SB}
=D+\tau\operatorname{div}\widehat b^+
 +\frac{\tau}{\gamma}\lVert\widehat b^+\rVert^2.
\]

All endpoint, factorization, diffusivity, horizon, conditioning, and `tau`
bounds are mandatory fields in
`configs/pilot_model/schrodinger_bridge.yaml`. The pilot rejects missing,
extra, or altered contract fields before training. Each dataset receives an
independent checkpoint. Candidate `tau` values are scored only on the
deterministic held-out subset of the source training split; the chosen index is
frozen before benchmark validation and test evaluation.

This canonical bridge is deliberately comparable to Gaussian diffusion:
setting \(\sigma^2=\gamma\tau\) gives the same population smoothing path and
the same full endpoint identity. Similar curves are therefore expected and do
not imply an implementation leak. An arbitrary fixed-source bridge, such as
\(\mathcal N(0,I)\to\mu_T\), would be a different experiment requiring a
versioned entropic-OT coupling and is not silently approximated here.

For a finite empirical terminal sample, the limit \(\tau\downarrow0\) is
atomic and ultimately returns zero at isolated training atoms. Results must be
interpreted in the mesoscopic regime selected from source-train rows, where the
Brownian bandwidth exceeds sample spacing while remaining below geometric
curvature scales.
