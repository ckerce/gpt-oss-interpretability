# Lab Notebook Excursions

This file captures side investigations and reboot-recovery notes that are relevant enough to preserve, but separate from the main repo notebook.

## 2026-03-11

### Excursion 1: Bregman Geometry / PLS / CASCADE Side Thread

`Source`
- Recovered from `/tmp/tmp.tmp` after the reboot interruption.

`Context`
- The interrupted thread suggested looking at `~/code/mechanistic_interpretability/` and the NeurIPS project directory to see what PLS and Bregman distance might contribute to the current line of work.
- The main concrete references mentioned were:
  - `downloads/2602.15293v1.pdf`
  - `downloads/NeurIPS_Hydra_Paper (2).pdf`
  - `companion-repo/neurips-2026-activation-clustering/analysis/bregman_geometry/`

`Recovered thesis`
- Park et al. / the Bregman-duality paper gives a strong explanation for why naive Euclidean activation steering can be geometrically incoherent at the softmax layer.
- The key type-error claim from the scratch note:
  - steering often adds a dual-space object directly in primal space
  - that is only coherent when primal and dual coordinates are effectively aligned
- The same note argued that the paper's scope is narrower than the clickbait framing:
  - the theory is cleanest at the softmax output layer
  - it does not by itself establish the correct geometry for arbitrary intermediate layers
  - practical steering papers mostly operate at intermediate layers, so the strongest claims should not be imported without qualification

`Recovered interpretation for PLS / CASCADE`
- The scratch note repeatedly converged on a stronger framing than "PLS helps dual steering."
- The more interesting interpretation was:
  - per-layer supervision in PLS / CASCADE may partially Euclideanize the usable intermediate-layer geometry
  - if so, Euclidean steering can work well not because the geometry problem disappears in theory, but because the architecture makes the primal-dual mismatch smaller in practice
- The note treated rising primal-dual alignment with depth, better Hessian conditioning, and improved effective rank as the relevant diagnostic signals.

`Recovered empirical claims from the scratch note`
- The note described an unexpected pattern:
  - in the PLS model, Euclidean steering reached target behavior faster than dual steering across several intermediate and late layers, with lower off-target KL
  - in the control model, dual steering had only a modest advantage in the layers where conditioning was worst
  - dual steering seemed mainly useful where both methods were already struggling
- The note interpreted this as evidence for:
  - architecture changes the geometry
  - better conditioning can reduce the practical need for dual correction
  - the interesting result may be "architectural Euclideanization" rather than "dual steering succeeds on PLS"

`What Bregman distance was proposed to do`
- Use Bregman / Fisher-style quantities as diagnostics, not as the headline result.
- Candidate uses recovered from the note:
  - characterize whether intermediate-layer prediction geometry is curved or approximately Euclidean
  - quantify whether PLS / CASCADE is flattening that geometry
  - measure primal-dual alignment, Hessian rank, and conditioning
  - explain why simple Euclidean steering may remain sound on interpretable-by-design architectures

`Caveats captured in the scratch note`
- Do not overstate the Park et al. result as if it proved the geometry of all intermediate layers.
- Do not make this the main NeurIPS arc yet.
- Treat the geometry story as:
  - a side note
  - an appendix candidate
  - or a future standalone note/paper
- The main repo arc should stay focused unless the geometry diagnostics become directly decision-relevant.

`Recovered action items`
- Keep the Bregman / PLS / CASCADE geometry investigation separate from the main notebook.
- If resumed, structure the excursion around:
  1. what Park et al. actually claim and where the assumptions stop
  2. how PLS / CASCADE may change those assumptions at intermediate layers
  3. which diagnostics matter: Hessian rank, condition number, primal-dual cosine, steering quality, off-target drift
  4. whether the right writeup target is a note, appendix, or standalone paper

`Status`
- Preserved here as an excursion memo.
- Not yet integrated into the main Entry 20 CASCADE-intervention bridge thread.
