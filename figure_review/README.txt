Replacement for Figure 1 (methodology overview)
===============================================

FILE:  Fig1_methodology_v2.png      3082 x 2739 px, 300 dpi
MADE BY: plot_methodology_v2.py  (in the repo root)

Nothing in the Overleaf/EMSE bundle has been touched. If you want this
figure, copy it over emse_submission/Fig1.png yourself and re-zip, or
upload it to Overleaf in place of Fig1.png.

WHY IT WAS REPLACED
-------------------
The original figure was built during the 30-sample pilot and never
refreshed. Five of its statements contradict the submitted manuscript:

   figure said                  manuscript says
   ---------------------------  ------------------------
   "first 30 used per cell"     all 100 functions
   "9 HumanEval + 21 MBPP"      35 HumanEval + 65 MBPP
   "480 cells"                  1,600 suites
   "n ~ 4-30 valid per cell"    n = 50-100
   "cosine top-3"               cosine top-5

It also had three text collisions (the track headings overlapped their
own body text) and type that was small for the canvas.

This slipped past check_paper_consistency.py because the figure-freshness
gate exempts "schematics", and this file was on that exemption list. The
exemption was wrong: the schematic carries data claims.

WHAT CHANGED IN THE DESIGN
--------------------------
 - Four numbered stages, one idea each, instead of an undifferentiated flow
 - The counts are the largest type on the page; prose is cut to short phrases
 - No overlapping text anywhere
 - Body type raised from ~9pt to 13.5-15pt equivalent; headline counts 22-23pt
 - The four techniques carry the same validated palette used by every other
   figure in the paper, so colours mean the same thing across figures
 - Statistical tests reduced from nine lines to one

EVERY NUMBER IS READ FROM THE ARTIFACTS AT BUILD TIME
-----------------------------------------------------
corpus_manifest.tsv, results_mutation.tsv and the analysis checkpoints are
read by the script; no count is typed into the figure. Re-running the script
after a data change updates the figure automatically, which is the failure
the original version had.

To regenerate:   python3 plot_methodology_v2.py --out <path>
