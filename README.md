# quantum-transport-maps

Official code for *Quantum variational parameters as classical data outputs*.

This repository studies a hybrid quantum-classical model in which a parameterized quantum circuit does not directly generate the final output through measurement. Instead, the circuit defines a bounded scalar compatibility score \(Q_\theta(x,y)\) for a source input \(x\) and a candidate target \(y\). The target is then recovered classically by optimizing \(y\) to maximize this learned score. In this view, the quantum model acts as a transport landscape or conditional scorer over pairs of classical data points.

The main idea of the paper is that variational quantum circuits can be used to learn structured maps between source and target domains by shaping the score along interpolating paths between them. Training encourages correct pairs and their transport trajectories to receive high score, while inference searches in data space for a target that best matches the source under the learned quantum score. This separates the role of the quantum circuit from the role of classical optimization: the circuit learns the landscape, and the optimizer extracts the output.

The repository includes low-dimensional transport experiments, latent-space applications, and small-scale QPU validation.

```md
## QPU scaling outlook

| Config | Qubits | Params | CZ gates | Gap (true-rand) | Correct | QPU viable? |
|---|---:|---:|---:|---:|---:|---|
| 2q r=2 | 2 | 34 | 10 | 0.014 | 8/20 | ✅ tested |
| 2q r=5 | 2 | 76 | 22 | 0.170 | 17/20 | ✅ tested |
| 4q r=3 | 4 | 96 | 28 | 0.039 | 9/20 | ❌ no signal |
| 4q r=5 | 4 | 152 | 44 | 0.131 | 15/20 | ✅ submit |
| 6q r=2 | 6 | 102 | 30 | 0.001 | 12/20 | ❌ no signal |
| 6q r=3 | 6 | 144 | 42 | 0.023 | 12/20 | ❌ barely |

The 6-qubit configurations do not discriminate strongly enough in simulation and would likely require \(r \ge 5\), corresponding to roughly 66 CZ gates. That pushes the circuit into a regime where hardware risk becomes much less attractive for a first verification run.

The strongest next QPU candidate is **4q / 2 copies / r=5**. It shows good simulated discrimination with a true-random gap of about 0.13 and correct ordering on 15 out of 20 test points, while remaining within a plausible hardware budget at 44 CZ gates. It also has an important structural advantage: it is the same multi-copy architecture used in the full Swiss Roll experiment, so a successful run would be a more meaningful step beyond the current 2-qubit validation.
```

## Theory justification

From a theoretical perspective, the model is best viewed as a bounded conditional scoring model rather than a direct quantum sampler. The score \(Q_\theta(x,y)\) is built from averaged Pauli-\(Z\) expectation values, so it is automatically bounded in \([-1,1]\). This boundedness makes the induced conditional density \(p_\theta(y \mid x) \propto e^{Q_\theta(x,y)}\) well behaved on compact domains and supports a MAP-style interpretation of inference.

The training objective does not learn a full probability distribution. Instead, it supervises the score along source-target interpolants and anchors the score of true pairs near its maximum. This gives a directional transport signal: along a fitted path from \(x\) to \(y\), the score is encouraged to increase monotonically toward the target. In practice, this means the learned quantum model can be interpreted as defining a potential-like landscape whose maxima correspond to valid outputs.

A second theoretical motivation comes from data reuploading. Repeated encoding layers increase the expressive frequency structure available to the variational circuit, making it more plausible that deeper reuploading can form sharper and more informative score landscapes. The practical question is therefore not only whether the circuit can represent the target map, but whether it learns a geometry in \(y\)-space that is smooth enough for optimization and sharp enough to distinguish correct targets from incorrect ones.

This repository explores that viewpoint empirically: the quantum circuit is treated as a learned bounded transport score, and the downstream classical optimizer converts that score into classical outputs.
