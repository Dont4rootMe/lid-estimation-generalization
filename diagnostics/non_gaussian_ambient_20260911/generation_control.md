# Supplemental generation check, declared before final weights and scores

At the roughly60000-step prefix, exact image posterior traces at low noise
still differed substantially from the known continuous posterior. This is
interim evidence, not a verdict on the final128000-step weights. Add a direct
generation check to distinguish useful denoising/LID scores from sample collapse.
It does not change training, model selection, tail settings or the LID grid.

Both native adapters imply the canonical ODE dy/dlog(lambda)=y-b. For t-Flow,
use x=t*y and t=k/(k+lambda) in its native velocity; for PFGM++ use r proportional
to lambda in its native radial field. Draw512 native Student initial noises at
lambda64, integrate512 Heun steps equally spaced in log(lambda) to1/256, then
take the learned posterior mean at that endpoint. No projection, teacher or
test object enters a trajectory. The finite initial scale approximates the
initial noisy marginal and is a stated limitation.

Repeat the first64 initial states with1024 steps. Record RMS difference per
coordinate in fit-normalized units; a value below.005 is the declared numerical
check, not an empirical quality threshold. Retain failed numerical checks and
the original sample set. Model or LID scale selection never uses these results.

Compare the generated distribution with all1000 canonical test observations:
total covariance trace ratio, mean shift, unbiased energy distance, and nearest
distance to the same8192-point fit-only bank. Distances use every supplied
ambient coordinate and are divided by the test distribution's RMS vector norm.
Also measure distance normal to the exact generator's affine span, solely as
an evaluation diagnostic. That span is never passed to the learned model or
sampler. Plot recovered generator coordinates together with this normal-error
number; a coordinate plot alone cannot certify a full-ambient sample.
