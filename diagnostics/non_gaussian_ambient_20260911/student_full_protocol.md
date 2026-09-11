# Supplementary native Student density readout, 2026-09-11 17:38 UTC

The eight-run primary response protocol remains frozen and unchanged. This
additional measurement was derived after inspecting two-query image training
prefixes; it is a mathematical extension, not a prospectively registered
quality repair. No completed 128k test result has been inspected when writing
this specification. It changes neither training nor model selection.

For the radial Student kernel with ambient dimension N and degrees of freedom
nu, define b(y,lambda) as its posterior mean, e=y-b, a=(nu-2)*lambda^2, and
R=trace(d_y b). The canonical scale lambda is per-coordinate noise RMS. The
density readout F=N+d_log(lambda)log rho_lambda(y) can be recovered as

    F = [R + (N+nu)*||e||^2/a + e.dot(d_log(lambda)b)/a]
        / [1+||e||^2/a].

All derivatives hold the canonical query y fixed. This is not the Gaussian
R+||e||^2/lambda^2 substitution. For an arbitrary approximate learned field,
the formula need not be a derivative of any normalized density; model-error
controls remain necessary.

One derivation uses the augmented Poisson field, r=lambda*sqrt(nu-2),
u=(y-b)/r. Continuity and equality of mixed derivatives give
(I+u u^T) grad_y log rho = d_r u - u*(div_y u+(nu-1)/r).
Contract with u, use div_y u=(N-R)/r, and substitute into
F=R-(y-b).grad_y log rho. This derivation also applies to radial Student
t-Flow because the conditional channel is the same kernel with its own nu.

Verify the formula against direct derivatives of exact finite Student mixtures
and against a continuous homogeneous half-line (whose density exponent is1,
while response alone is deficient). Verify the extra scale derivative on
actual trained vector and U-Net models with central finite differences.

For each of the same eight final checkpoints, reuse the unchanged response
trace arrays/probes. Add the deterministic scale derivative using all original
ambient coordinates. Evaluate all29 scales on the same1000 holdout queries,
select by the same mean absolute LID error, and persist/hash that selection
before computing this readout on test. Use all1000 canonical test queries.
Save all arrays, source/weight/primary-selection hashes, and selected metrics.
Original response receipts and measurements are never overwritten. Report
the new readout as supplementary and do not choose between R and F on test.
