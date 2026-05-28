# Cover Letter — Software Quality Journal Submission

**To:** The Editors, *Software Quality Journal*
**From:** Balaji Venktesh, [advisor], …
**Re:** Submission of *Mutation-Testing Quality of LLM- and RAG-based Unit-Test Generators: A Cross-Model Empirical Study*

---

Dear Editors,

We submit for your consideration the enclosed manuscript, *Mutation-
Testing Quality of LLM- and RAG-based Unit-Test Generators: A
Cross-Model Empirical Study*. The paper reports a
cross-method, cross-LLM, cross-benchmark empirical evaluation of
LLM-based unit-test generation, evaluated through the lens of
**mutation kill rate** — a behavioural defect-detection metric that
sits squarely within the *Software Quality Journal*'s historical
scope.

The paper makes five contributions that we believe will be of
direct interest to your readership:

1. **A 4 × 4 mutation-testing matrix** comparing four generation
   methods (Plain LLM, Random RAG, Simple RAG, Iterative Critique
   RAG) across four open-weight LLMs spanning 3 B – 30 B parameters
   and dense vs mixture-of-experts architectures, on a 100-sample
   subset of HumanEval and MBPP. To our knowledge no prior study
   reports this matrix.

2. **Statistically rigorous analysis** using Type-III ANOVA,
   mixed-effects regression with `sample_idx` as a random intercept,
   Tukey HSD post-hoc, and per-operator + per-benchmark
   decompositions. We identify a statistically significant effect of
   Iterative Critique RAG over Plain LLM specifically on **boundary
   mutators in MBPP-style numeric problems** (Tukey HSD ∆ = +0.31,
   p_adj = 0.025; Mixed-LM Wald-test p = 0.005), a finding that does
   not surface in pooled analyses.

3. **A three-annotator human-evaluation study** under a
   behaviourally-anchored 0–5 rubric covering test idiom quality,
   correctness, and completeness. Iterative Critique RAG ranks
   highest on all three dimensions, replicating the mutation-testing
   method ordering with independent human evidence.

4. **A head-to-head comparison against Pynguin 0.45.0**, the
   reference search-based test generator for Python, on the same 40
   functions used in the human evaluation. LLM methods outperform
   Pynguin's SBST on overall mutation kill rate but the gap is
   concentrated on the comparison-operator family (Pynguin 0.33 vs
   IC 0.96), suggesting SBST and LLM-based generators are
   complementary on different operator families.

5. **A counterintuitive negative-faithfulness finding**:
   token-overlap faithfulness between generated tests and retrieved
   documentation negatively predicts mutation kill rate (Pearson
   r = −0.61, p = 0.045). RAG pipelines that reward syntactic
   faithfulness optimise the wrong objective.

The paper is approximately 14,500 words. We have prepared figures,
tables, and a publicly-accessible replication package on GitHub. All
empirical results are reproducible from the released artefacts.

We confirm that this manuscript is original work, has not been
previously published, and is not under consideration for publication
elsewhere. All authors have read and approved the submission. We have
no conflicts of interest to declare.

We propose the following potential reviewers based on their
publication history in the related areas (any of whom we would be
delighted to have review the work):

- [Reviewer 1 name — affiliation — expertise]
- [Reviewer 2 name — affiliation — expertise]
- [Reviewer 3 name — affiliation — expertise]

We thank you for considering our submission, and we look forward to
the editorial decision.

With kind regards,

Balaji Venktesh
[Affiliation, address, email]

---

> **Author note (not for submission):** Fill in the bracketed
> reviewer suggestions before sending. Two or three names from
> mutation-testing or test-generation research who are NOT direct
> collaborators of the authors. Candidates worth considering:
> a Pynguin co-author (Stephan Lukasczyk, Univ. of Passau), a
> mutation-testing methodologist (Mike Papadakis, Univ. of
> Luxembourg), an LLM-test-generation author (Max Schäfer,
> GitHub Next, if appropriate for journal-track review).
