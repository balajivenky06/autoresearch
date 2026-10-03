```

==========================================================================
 1. CORPUS AND TOTALS   (section 3.2, section 4 preamble)
==========================================================================
  functions in corpus        : 100
  corpus fingerprint         : ccd19ffcb6092603
  generation cells           : 16  (4 methods x 4 models)
  valid per-sample observations: 1443
  mutants generated          : 9660
  mutants killed             : 7668
  equivalent mutants         : 619 (6.4% of mutants)
  benchmark split            : {'mbpp': 65, 'humaneval': 35}

==========================================================================
 2. PER-CELL KILL RATE MATRIX   (Table 2, appendix per-cell table)
==========================================================================
  method                       llama3.2             phi4          qwen3.5      qwen3-coder
  Plain LLM            0.7436 (n= 95) 0.8969 (n=100) 0.9285 (n=100) 0.9261 (n=100) 
  Random RAG           0.7061 (n= 80) 0.9092 (n= 96) 0.9598 (n= 99) 0.9429 (n= 99) 
  Simple RAG           0.7246 (n= 87) 0.9157 (n= 94) 0.9594 (n= 96) 0.9293 (n=100) 
  Iterative Critique   0.6963 (n= 50) 0.9092 (n= 71) 0.9514 (n= 77) 0.9293 (n= 99) 

  column mean (all 4 cells):
                                 0.7177           0.9078           0.9498           0.9319 

==========================================================================
 3. PER-MODEL AND PER-METHOD MEANS   (section 4.1)
==========================================================================
  per-method (averaged across models), best first:
    Simple RAG           0.8823
    Random RAG           0.8795
    Plain LLM            0.8738
    Iterative Critique   0.8716
  per-model (averaged across methods), weakest first:
    llama3.2_latest      0.7177
    phi4_14b             0.9078
    qwen3-coder_30b      0.9319
    qwen3.5_9b           0.9498

  method spread : 0.0107
  model spread  : 0.2321
  ratio         : 21.7x  <- headline for section 4.1

==========================================================================
 4. SWINGS   (section 4.1 — the reviewer's Table 2 objection)
==========================================================================
  within-model swing, llama3.2_latest      0.0473
  within-model swing, phi4_14b             0.0188
  within-model swing, qwen3.5_9b           0.0313
  within-model swing, qwen3-coder_30b      0.0168

  largest within-model swing : 0.0473
  mean within-model swing    : 0.0285
  cross-model gap            : 0.2321
  cross/within ratio         : 4.9x
  -> the claim holds outright; no cell exclusion needed

==========================================================================
 5. STATISTICS   (section 4.2, Table 3)
==========================================================================
  Type-III ANOVA on kill rate:
    C(method)        sum_sq=   0.020  df=   3  F=    0.209  p=0.8904
    C(model)         sum_sq=  10.383  df=   3  F=  108.543  p=6.752e-63
    C(sample_idx)    sum_sq=  31.443  df=  99  F=    9.961  p=2.74e-103
    Residual         sum_sq=  42.631  df=1337
  Tukey HSD significant method pairs: 0/6
  Kruskal-Wallis: H=0.8198  p=0.8447
  Friedman (paired): chi2=2.9174  p=0.4045  blocks=273
  Mixed-LM (sample_idx random intercept):
    C(method)[T.plain_llm]                   beta=+0.0005  p=0.9722
    C(method)[T.random_rag]                  beta=+0.0079  p=0.5739
    C(method)[T.simple_rag]                  beta=+0.0076  p=0.5842
    C(model)[T.phi4_14b]                     beta=+0.1831  p=1.113e-39
    C(model)[T.qwen3-coder_30b]              beta=+0.2073  p=2.228e-52
    C(model)[T.qwen3.5_9b]                   beta=+0.2245  p=1.214e-59
    group variance 0.0198

==========================================================================
 6. PER-BENCHMARK   (section 4.2.2, Table 4)
==========================================================================
  humaneval  kill_rate            n=  498  F=  0.584  p=0.6256   Tukey IC-vs-Plain delta=+0.0172 p_adj=0.8683
  humaneval  kill_rate_boundary   n=  403  F=  0.488  p=0.6910   Tukey IC-vs-Plain delta=-0.0166 p_adj=0.9682
  mbpp       kill_rate            n=  945  F=  0.340  p=0.7963   Tukey IC-vs-Plain delta=-0.0337 p_adj=0.5595
  mbpp       kill_rate_boundary   n=  521  F=  0.941  p=0.4206   Tukey IC-vs-Plain delta=-0.0900 p_adj=0.2160
  pooled     kill_rate            n= 1443  F=  0.209  p=0.8904   Tukey IC-vs-Plain delta=-0.0156 p_adj=0.8381
  pooled     kill_rate_boundary   n=  924  F=  0.615  p=0.6057   Tukey IC-vs-Plain delta=-0.0578 p_adj=0.2401

==========================================================================
 7. PER-OPERATOR   (section 4.4)
==========================================================================
  method                arithmeti   boundary  compariso  negate_bo  return_no
  Plain LLM                0.7615     0.7546     0.8626     0.8111     0.9252 
  Random RAG               0.7796     0.7596     0.8592     0.8272     0.9448 
  Simple RAG               0.7743     0.7754     0.8610     0.8512     0.9442 
  Iterative Critique       0.7831     0.8123     0.8620     0.7924     0.9269 

==========================================================================
 8. ATTRITION AND MATCHED SUBSET   (new; answers the IC confound)
==========================================================================
  model                  Plain LLM  Random RAG  Simple RAG  Iterative    all-four
  llama3.2_latest               95          80          87          50         38
  phi4_14b                     100          96          94          71         64
  qwen3.5_9b                   100          99          96          77         73
  qwen3-coder_30b              100          99         100          99         98

  matched functions pooled: 273
    Random RAG           0.9062
    Iterative Critique   0.8967
    Simple RAG           0.8956
    Plain LLM            0.8954
  matched method spread   : 0.0108   (unmatched 0.0107)
  Friedman on matched     : chi2=2.9174 p=0.4045
  -> attrition does not confound the method comparison

==========================================================================
 9. DECONTAMINATION   (new; answers the contamination objection)
==========================================================================
  functions 98   observations 1421   mutants 9404   killed 7533
  method               model                  main    decon    delta
  Plain LLM            llama3.2_latest      0.7436   0.7648  +0.0211
  Plain LLM            phi4_14b             0.8969   0.9088  +0.0119
  Plain LLM            qwen3.5_9b           0.9285   0.9630  +0.0345
  Plain LLM            qwen3-coder_30b      0.9261   0.9266  +0.0005
  Random RAG           llama3.2_latest      0.7061   0.7524  +0.0463
  Random RAG           phi4_14b             0.9092   0.9381  +0.0290
  Random RAG           qwen3.5_9b           0.9598   0.9729  +0.0131
  Random RAG           qwen3-coder_30b      0.9429   0.9295  -0.0134
  Simple RAG           llama3.2_latest      0.7246   0.7513  +0.0267
  Simple RAG           phi4_14b             0.9157   0.9322  +0.0165
  Simple RAG           qwen3.5_9b           0.9594   0.9401  -0.0194
  Simple RAG           qwen3-coder_30b      0.9293   0.9302  +0.0009
  Iterative Critique   llama3.2_latest      0.6963   0.7666  +0.0704
  Iterative Critique   phi4_14b             0.9092   0.8970  -0.0123
  Iterative Critique   qwen3.5_9b           0.9514   0.9370  -0.0144
  Iterative Critique   qwen3-coder_30b      0.9293   0.9504  +0.0211

  mean delta   +0.0145
  median delta +0.0148
  range        -0.0194 .. +0.0704
  cells up     12/16
  decontaminated ANOVA: method F=0.072 p=0.9751   model F=82.920 p=3.846e-49
  decontaminated method spread 0.0105   (main 0.0107)  -> null replicates on a renamed corpus

==========================================================================
 10. CEILING EFFECT   (section 3.4 threat, now quantified)
==========================================================================
  exactly 0.0            :    40  (2.8%)
  exactly 1.0            :  1056  (73.2%)
  strictly between       :   347  (24.0%)
  -> justifies the rank-test-first strategy; cite these figures rather
     than asserting a ceiling effect
```
