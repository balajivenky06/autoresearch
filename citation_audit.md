# Citation audit

- Entries: **51**
- Blockers: **0** · Warnings: **20** · Clean: **31**
- Entries never \cite'd in `paper_draft.tex`: **0**
- Mode: Crossref-verified

## Warnings

### `lemieux2023codamosa` — DESCRIPTION_SUSPECT
- bib title: CodaMosa: Escaping Coverage Plateaus in Test Generation with Pre-Trained Large Language Models
- crossref title: **CodaMosa: Escaping Coverage Plateaus in Test Generation with Pre-trained Large Language Models**
- doi: `10.1109/ICSE48619.2023.00085`  · cited 3x
- registry venue: 2023 IEEE/ACM 45th International Conference on Software Engineering (ICSE)
- registry type: proceedings-article
- prose near \cite claims 'sbst', 'search-based' but real title is 'CodaMosa: Escaping Coverage Plateaus in Test Generation with Pre-trained Large Language Models'
  - > …om docstrings or function signatures, and require no per-function search budget beyond inference time \citep{schafer2024,tufano2022}. Multiple recent studies have benchmarked LLM-generated tests against search-based baselines and reported competitive or superior effectiveness on standard benchmarks \citep{schafer2024,lemieux2023codamosa,dakhel2024mutation, straubinger2025debug,konstantinou2026newer}. The question that frames the present paper is therefore no longer ``can LLMs replace search-based generation?''--- the empirical answer to that is `…
  - > …e models; the training data came from \citet{tufano2022}'s Methods2Test, a corpus of focal methods mapped to their test cases, which is a dataset contribution rather than a model. The arrival of decoder-only frontier LLMs (Codex, GPT-3.5, GPT-4) shifted research toward prompt-based test generation. \citet{lemieux2023codamosa} combined LLM prompting with search-based fallbacks, using the LLM to escape coverage plateaus where pure SBST runs got stuck. \citet{schafer2024} int…
  - > …typing and runtime introspection capabilities. The Pynguin authors and others have benchmarked it on standard Python benchmarks (HumanEval, MBPP) and reported competitive coverage results against test-suite generation baselines. \paragraph{Empirical SBST-vs-LLM comparisons} are recent and limited. \citet{lemieux2023codamosa} is the closest analog: it combines SBST with LLM prompts and reports improvements over each approach alone, demonstrating complementarity in coverage…

### `schafer2024` — NO_DOI, DESCRIPTION_SUSPECT
- bib title: An Empirical Evaluation of Using Large Language Models for Automated
             Unit Test Generation
- doi: `(none)`  · cited 5x
- prose near \cite claims 'search-based' but real title is 'An Empirical Evaluation of Using Large Language Models for Automated
             Unit Test Gen'
  - > …readable test code has shifted this landscape. LLMs trained on public source code can produce pytest- or JUnit-formatted test suites that read like hand-written tests, encode specifications drawn from docstrings or function signatures, and require no per-function search budget beyond inference time \citep{schafer2024,tufano2022}. Multiple recent studies have benchmarked LLM-generated tests against search-based baselines and reported competitive or superior effectiveness on st…
  - > …om docstrings or function signatures, and require no per-function search budget beyond inference time \citep{schafer2024,tufano2022}. Multiple recent studies have benchmarked LLM-generated tests against search-based baselines and reported competitive or superior effectiveness on standard benchmarks \citep{schafer2024,lemieux2023codamosa,dakhel2024mutation, straubinger2025debug,konstantinou2026newer}. The question that frames the present paper is therefore no longer ``can LLMs replace search-based generation?''--- the empirical answer to that is `…
  - > …tion rather than a model. The arrival of decoder-only frontier LLMs (Codex, GPT-3.5, GPT-4) shifted research toward prompt-based test generation. \citet{lemieux2023codamosa} combined LLM prompting with search-based fallbacks, using the LLM to escape coverage plateaus where pure SBST runs got stuck. \citet{schafer2024} introduced an adaptive generation loop in which the LLM iteratively refines its tests against runtime feedback, and provides the largest empirical co…

### `lewis2020` — DESCRIPTION_SUSPECT
- bib title: Retrieval-augmented generation for knowledge-intensive NLP tasks
- crossref title: **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks**
- doi: `10.48550/arXiv.2005.11401`  · cited 3x
- resolved via DataCite (preprint / non-Crossref registrant)
- registry venue: arXiv
- registry type: Preprint
- prose near \cite claims 'test-generation' but real title is 'Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks'
  - > …d the dominant augmentation is retrieval. We introduce that idea first, then state the question this paper asks about it. \subsection{Retrieval augmentation for test generation}\label{sec:intro-rag} The augmentation we study is \emph{retrieval-augmented generation} (RAG). The original RAG framework \citep{lewis2020} augments an LLM's prompt with passages retrieved from a knowledge base; in the test-generation context, the knowledge base is typically a curated set…
  - > …mplementation, as the prior-paradigm reference. SBST tools derive assertions from \emph{observed} return values, which is the source of the \emph{regression oracle problem}: the assertions encode what the program currently does, not what it should do. \paragraph{Retrieval-augmented generation.} RAG \citep{lewis2020} prepends material retrieved from a corpus to a model's prompt. Retrieval here is \emph{dense}: both the query and each corpus chunk are embedded into…
  - > …two results bound the question our study sits inside: generic document retrieval appears to add little, while task-specific structural retrieval appears to add a great deal. Our knowledge base is of the first kind, and our null should be read as evidence about that kind. The original RAG framework \citep{lewis2020} demonstrated that augmenting a sequence-generation LLM with a passage-retrieval step produces better outputs on knowledge-intensive NLP tasks. The fr…

### `zhang2023repocoder` — DESCRIPTION_SUSPECT
- bib title: {R}epo{C}oder: Repository-Level Code Completion Through Iterative Retrieval and Generation
- crossref title: **RepoCoder: Repository-Level Code Completion Through Iterative Retrieval and Generation**
- doi: `10.18653/v1/2023.emnlp-main.151`  · cited 2x
- registry venue: Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing
- registry type: proceedings-article
- prose near \cite claims 'test generation' but real title is 'RepoCoder: Repository-Level Code Completion Through Iterative Retrieval and Generation'
  - > …trast against Simple RAG isolates the contribution of retrieval \emph{relevance} from that of simply having more context in the prompt. Studies that omit this arm cannot separate the two. \end{itemize} \noindent Retrieval has been evaluated on adjacent code tasks such as repository-level completion \citep{zhang2023repocoder} and retrieval-augmented code generation \citep{su2025evor,lu2022reacc}. For \emph{unit test generation specifically} the literature is thinner: \cite…
  - > …ion contexts. \citet{parvez2021} showed that retrieval-augmented code summarization and generation could improve both code-completion and natural-language-to-code translation. \citet{lu2022reacc} demonstrated a retrieval-augmented code-completion framework using both lexical and semantic retrieval. \citet{zhang2023repocoder} introduced iterative retrieval at the repository level, where retrieval is re-run after each draft refinement —conceptually similar to our Iterative…

### `demillo1978` — DESCRIPTION_SUSPECT
- bib title: Hints on Test Data Selection: Help for the Practicing Programmer
- crossref title: **Hints on Test Data Selection: Help for the Practicing Programmer**
- doi: `10.1109/C-M.1978.218136`  · cited 2x
- registry venue: Computer
- registry type: journal-article
- prose near \cite claims 'mutation testing' but real title is 'Hints on Test Data Selection: Help for the Practicing Programmer'
  - > …ribution at their intersection. \subsection{Background and terminology}\label{sec:related-background} This subsection fixes the terms used throughout. Readers familiar with mutation testing and retrieval augmentation may skip to \S\ref{sec:related-llmtg}. \paragraph{Mutation testing.} Introduced by \citet{demillo1978}, mutation testing evaluates a test suite by injecting small, syntactically valid faults into the program under test and asking whether the suite noti…
  - > …We use rule-based AST operators rather than LLM-generated mutants for the same reason---\citet{wang2026mutationstudy} show the latter are more realistic, but they would introduce a second model-dependent factor into a design whose purpose is to isolate the first. Mutation testing was introduced by \citet{demillo1978} as a thought experiment about test-adequacy and was operationalized over the next 30 years into a workable empirical methodology. The foundational em…

### `andrews2005` — DESCRIPTION_SUSPECT
- bib title: Is Mutation an Appropriate Tool for Testing Experiments?
- crossref title: **Is mutation an appropriate tool for testing experiments?**
- doi: `10.1145/1062455.1062530`  · cited 6x
- registry venue: Proceedings of the 27th international conference on Software engineering  - ICSE '05
- registry type: proceedings-article
- prose near \cite claims 'sbst' but real title is 'Is mutation an appropriate tool for testing experiments?'
  - > …ion capability if it asserts only structural properties (return type, list length) rather than specific oracle values. The SE-relevant operationalization of ``do the tests catch bugs?'' is \emph{the mutation kill rate}—the fraction of systematically-injected code defects that the test suite detects \citep{andrews2005,just2014}. Mutation testing has been a gold-standard metric in the SBST literature for two decades but has been used only sporadically in LLM-test-generation e…
  - > …numeric boundary. Mutation testing is used here as an \emph{evaluation} metric; much recent work instead uses it \emph{generatively}, feeding surviving mutants back into the prompt (\S\ref{sec:related-mutation}). The metric earns its place by being a better proxy for fault detection than coverage. \citet{andrews2005} and \citet{just2014} showed that mutant detection correlates with real-fault detection more strongly than coverage does, which matters here because a…
  - > …tion testing was introduced by \citet{demillo1978} as a thought experiment about test-adequacy and was operationalized over the next 30 years into a workable empirical methodology. The foundational empirical justification—that mutation score correlates with real fault detection capability—came from \citet{andrews2005}, who showed that detection rates of injected mutants are statistically correlated with detection rates of real faults from project bug-tracker histor…

### `just2014` — DESCRIPTION_SUSPECT
- bib title: Are Mutants a Valid Substitute for Real Faults in Software Testing?
- crossref title: **Are mutants a valid substitute for real faults in software testing?**
- doi: `10.1145/2635868.2635929`  · cited 5x
- registry venue: Proceedings of the 22nd ACM SIGSOFT International Symposium on Foundations of Software Engineering
- registry type: proceedings-article
- prose near \cite claims 'sbst' but real title is 'Are mutants a valid substitute for real faults in software testing?'
  - > …ion capability if it asserts only structural properties (return type, list length) rather than specific oracle values. The SE-relevant operationalization of ``do the tests catch bugs?'' is \emph{the mutation kill rate}—the fraction of systematically-injected code defects that the test suite detects \citep{andrews2005,just2014}. Mutation testing has been a gold-standard metric in the SBST literature for two decades but has been used only sporadically in LLM-test-generation e…
  - > …ion testing is used here as an \emph{evaluation} metric; much recent work instead uses it \emph{generatively}, feeding surviving mutants back into the prompt (\S\ref{sec:related-mutation}). The metric earns its place by being a better proxy for fault detection than coverage. \citet{andrews2005} and \citet{just2014} showed that mutant detection correlates with real-fault detection more strongly than coverage does, which matters here because a suite can reach full…
  - > …odology. The foundational empirical justification—that mutation score correlates with real fault detection capability—came from \citet{andrews2005}, who showed that detection rates of injected mutants are statistically correlated with detection rates of real faults from project bug-tracker history. \citet{just2014} provided a follow-up large-scale study on Java projects that confirmed the result. The mutation-testing tool ecosystem includes \textit{PIT} for Java…

### `papadakis2019survey` — DESCRIPTION_SUSPECT
- bib title: Mutation Testing Advances: An Analysis and Survey
- crossref title: **Mutation Testing Advances: An Analysis and Survey**
- doi: `10.1016/bs.adcom.2018.03.015`  · cited 4x
- registry venue: Advances in Computers
- registry type: book-chapter
- prose near \cite claims 'sbst' but real title is 'Mutation Testing Advances: An Analysis and Survey'
  - > …nt} when the injected change leaves observable behaviour unaltered---for instance replacing \verb|x < n| with \verb|x != n| inside a loop that increments \verb|x| by one---so that no test can kill it; equivalent mutants are excluded from the denominator, and detecting them is undecidable in general \citep{papadakis2019survey}. A \emph{mutation operator} is the rule that generates a class of mutants, such as replacing an arithmetic operator or perturbing a numeric boundary.…
  - > …rojects that confirmed the result. The mutation-testing tool ecosystem includes \textit{PIT} for Java \citep{coles2016} and \textit{mutmut} for Python. Our mutation operators (arithmetic, comparison, boundary, return-replacement, boolean-negation) are the canonical subset implemented by both tools. \citet{papadakis2019survey} provides the canonical recent survey of mutation testing, including the equivalent-mutant detection challenge that we address via ground-truth tests…
  - > …e provide here. The closest comparison is \citet{wang2026mutationstudy}, which reports the mutation kill rate as one of several evaluation metrics in a benchmark of LLMs-generated tests; their study covers fewer LLMs than ours and does not decompose the kill rate by operator or by source benchmark. \citet{papadakis2019survey} surveys the mutation-testing literature and catalogues the operator families that later work builds on, motivating the kind of operator-level analysi…

### `fraser2011` — DESCRIPTION_SUSPECT
- bib title: EvoSuite: automatic test suite generation for object-oriented software
- crossref title: **EvoSuite**
- doi: `10.1145/2025113.2025179`  · cited 3x
- registry stores an abbreviated title ('EvoSuite'); verified by hand against authors, venue and year
- registry venue: Proceedings of the 19th ACM SIGSOFT symposium and the 13th European conference on Foundations of software engi
- registry type: proceedings-article
- prose near \cite claims 'coverage', 'sbst', 'search-based', 'test generation' but real title is 'EvoSuite'
  - > …2017}. The dominant paradigm prior to 2022 was \emph{search-based software testing} (SBST): tools like EvoSuite for Java and Pynguin for Python treat test-suite synthesis as an optimization problem, evolving a population of candidate test cases against a coverage- or mutation-based fitness function \citep{fraser2011,lukasczyk2022}. These tools achieve high branch coverage on self-contained functions and have demonstrated practical value in industrial deployments, but they suffe…
  - > …suite can reach full statement coverage while asserting almost nothing. \paragraph{Search-based software testing (SBST).} The dominant pre-LLM paradigm treats test generation as an optimisation problem, evolving a population of candidate inputs against a coverage- or mutation-based fitness function \citep{mcminn2004,fraser2011}. We use Pynguin \citep{lukasczyk2022}, the standard Python implementation, as the prior-paradigm reference. SBST tools derive assertions from \emph{o…
  - > …\subsection{Search-based software testing}\label{sec:related-sbst} Search-based software testing has been the dominant paradigm for automated test-suite generation since \citet{mcminn2004}'s survey and \citet{harman2010}'s empirical comparison of search-based versus random testing. \emph{EvoSuite} \citep{fraser2011,fraser2013} is the canonical SBST tool for Java, combining genetic-algorithm test-case search with dynamic symbolic execution. EvoSuite has been validated repeat…

### `fraser2013` — DESCRIPTION_SUSPECT
- bib title: Whole Test Suite Generation
- crossref title: **Whole Test Suite Generation**
- doi: `10.1109/TSE.2012.14`  · cited 1x
- registry venue: IEEE Transactions on Software Engineering
- registry type: journal-article
- prose near \cite claims 'sbst' but real title is 'Whole Test Suite Generation'
  - > …\subsection{Search-based software testing}\label{sec:related-sbst} Search-based software testing has been the dominant paradigm for automated test-suite generation since \citet{mcminn2004}'s survey and \citet{harman2010}'s empirical comparison of search-based versus random testing. \emph{EvoSuite} \citep{fraser2011,fraser2013} is the canonical SBST tool for Java, combining genetic-algorithm test-case search with dynamic symbolic execution. EvoSuite has been validated repeat…

### `almasi2017` — DESCRIPTION_SUSPECT
- bib title: An Industrial Evaluation of Unit Test Generation: Finding Real
               Faults in a Financial Application
- crossref title: **An Industrial Evaluation of Unit Test Generation: Finding Real Faults in a Financial Application**
- doi: `10.1109/ICSE-SEIP.2017.27`  · cited 2x
- registry venue: 2017 IEEE/ACM 39th International Conference on Software Engineering: Software Engineering in Practice Track (I
- registry type: proceedings-article
- prose near \cite claims 'kill rate', 'mutation testing', 'sbst', 'search-based' but real title is 'An Industrial Evaluation of Unit Test Generation: Finding Real Faults in a Financial Applicatio'
  - > …\end{keyword} \end{frontmatter} \section{Introduction}\label{sec:introduction} Automated unit-test generation has been a target of empirical software-engineering research for decades, motivated by the well-documented cost of manual test authoring and the high marginal value of each additional test \citep{daka2014,almasi2017}. The dominant paradigm prior to 2022 was \emph{search-based software testing} (SBST): tools like EvoSuite for Java and Pynguin for Python treat test-…
  - > …t{harman2010}'s empirical comparison of search-based versus random testing. \emph{EvoSuite} \citep{fraser2011,fraser2013} is the canonical SBST tool for Java, combining genetic-algorithm test-case search with dynamic symbolic execution. EvoSuite has been validated repeatedly on industrial codebases \citep{almasi2017} and remains the reference baseline for Java-language SBST research. For Python, the corresponding tool is \emph{Pynguin} \citep{lukasczyk2023empirica…

### `lukasczyk2022` — NO_DOI, DESCRIPTION_SUSPECT
- bib title: Pynguin: Automated Unit Test Generation for {P}ython
- doi: `(none)`  · cited 5x
- prose near \cite claims 'sbst', 'search-based' but real title is 'Pynguin: Automated Unit Test Generation for {P}ython'
  - > …2017}. The dominant paradigm prior to 2022 was \emph{search-based software testing} (SBST): tools like EvoSuite for Java and Pynguin for Python treat test-suite synthesis as an optimization problem, evolving a population of candidate test cases against a coverage- or mutation-based fitness function \citep{fraser2011,lukasczyk2022}. These tools achieve high branch coverage on self-contained functions and have demonstrated practical value in industrial deployments, but they suffe…
  - > …asserting almost nothing. \paragraph{Search-based software testing (SBST).} The dominant pre-LLM paradigm treats test generation as an optimisation problem, evolving a population of candidate inputs against a coverage- or mutation-based fitness function \citep{mcminn2004,fraser2011}. We use Pynguin \citep{lukasczyk2022}, the standard Python implementation, as the prior-paradigm reference. SBST tools derive assertions from \emph{observed} return values, which is the s…
  - > …BST tool for Java, combining genetic-algorithm test-case search with dynamic symbolic execution. EvoSuite has been validated repeatedly on industrial codebases \citep{almasi2017} and remains the reference baseline for Java-language SBST research. For Python, the corresponding tool is \emph{Pynguin} \citep{lukasczyk2023empirical,lukasczyk2022}. Pynguin combines coverage-driven genetic search with dynamic symbolic execution, optimized for Python's dynamic typing and runtime introspection cap…

### `daka2014` — DESCRIPTION_SUSPECT
- bib title: A Survey on Unit Testing Practices and Problems
- crossref title: **A Survey on Unit Testing Practices and Problems**
- doi: `10.1109/ISSRE.2014.11`  · cited 1x
- registry venue: 2014 IEEE 25th International Symposium on Software Reliability Engineering
- registry type: proceedings-article
- prose near \cite claims 'kill rate', 'mutation testing', 'search-based' but real title is 'A Survey on Unit Testing Practices and Problems'
  - > …\end{keyword} \end{frontmatter} \section{Introduction}\label{sec:introduction} Automated unit-test generation has been a target of empirical software-engineering research for decades, motivated by the well-documented cost of manual test authoring and the high marginal value of each additional test \citep{daka2014,almasi2017}. The dominant paradigm prior to 2022 was \emph{search-based software testing} (SBST): tools like EvoSuite for Java and Pynguin for Python treat test-…

### `landis1977` — DESCRIPTION_SUSPECT
- bib title: The Measurement of Observer Agreement for Categorical Data
- crossref title: **The Measurement of Observer Agreement for Categorical Data**
- doi: `10.2307/2529310`  · cited 4x
- registry venue: Biometrics
- registry type: journal-article
- prose near \cite claims 'humaneval' but real title is 'The Measurement of Observer Agreement for Categorical Data'
  - > …me artefacts we report Cohen's $\kappa$ for pairs and Krippendorff's $\alpha$ \citep{krippendorff2018} across all three, both of which correct observed agreement for the agreement expected by chance. Values run from $-1$ to $1$; $\alpha \geq 0.4$ is the conventional floor for ``moderate'' agreement \citep{landis1977}. The \emph{ordinal} variant of $\alpha$, which we use, weights disagreements by how far apart the two ratings are on the scale. \subsection{LLM-based…
  - > …we present in \S\ref{sec:results-humaneval}. \paragraph{Rating-scale methodology.} The behaviorally-anchored rating scale (BARS) methodology we use in our rubric was introduced by \citet{smith1963} in industrial-psychology research and has been adapted for many software-engineering contexts since. \citet{landis1977} provides the canonical interpretation thresholds for Cohen's $\kappa$ that we use to assess inter-rater agreement in \S\ref{sec:results-humaneval}: $…
  - > …g analysis with a developer-perceived quality signal, we designed a three-dimension, behaviourally-anchored rating scale and ran a three-annotator study against it. The rubric is our own; the behaviourally-anchored format follows \citet{smith1963}, and we interpret agreement using the thresholds of \citet{landis1977} with ordinal Krippendorff's $\alpha$ \citep{krippendorff2018} as the primary three-rater statistic. \paragraph{Sample selection.} We drew 40 stratifi…

### `zar1984` — NO_DOI
- bib title: Biostatistical Analysis
- doi: `(none)`  · cited 1x
  - > …erroni correction across the family of pairwise tests—follows the standard recommendations of \citet{wohlin2012} and \citet{madeyski2024empirical} for analyzing empirical software-engineering experiments. The Spearman $\rho$ threshold of $\geq 0.8$ for cross-condition generalization is sourced from \citet{zar1984} and is the threshold adopted by \citet{jureczko2015} for defect-prediction-model generalization across projects, which is the SE literature's nearest…

### `krippendorff2018` — NO_DOI
- bib title: Content Analysis: An Introduction to Its Methodology
- doi: `(none)`  · cited 4x
  - > …how far the generated output reflects the retrieved material; we measure it both lexically and with a model judge (\S\ref{sec:methods-faithfulness}). \paragraph{Agreement statistics.} Where multiple annotators rate the same artefacts we report Cohen's $\kappa$ for pairs and Krippendorff's $\alpha$ \citep{krippendorff2018} across all three, both of which correct observed agreement for the agreement expected by chance. Values run from $-1$ to $1$; $\alpha \geq 0.4$ is th…
  - > …anonical interpretation thresholds for Cohen's $\kappa$ that we use to assess inter-rater agreement in \S\ref{sec:results-humaneval}: $\kappa < 0.20$ slight, $\kappa \in [0.21, 0.40]$ fair, $\kappa \in [0.41, 0.60]$ moderate, $\kappa \in [0.61, 0.80]$ substantial, $\kappa \geq 0.81$ almost perfect. \citet{krippendorff2018} defines the ordinal-$\alpha$ variant of inter-rater agreement we use as the primary three-rater statistic since Cohen's $\kappa$ is defined only pair…
  - > …designed a three-dimension, behaviourally-anchored rating scale and ran a three-annotator study against it. The rubric is our own; the behaviourally-anchored format follows \citet{smith1963}, and we interpret agreement using the thresholds of \citet{landis1977} with ordinal Krippendorff's $\alpha$ \citep{krippendorff2018} as the primary three-rater statistic. \paragraph{Sample selection.} We drew 40 stratified \texttt{(function, generated\_tests)} pairs from the mutati…

### `dakhel2024mutation` — DESCRIPTION_SUSPECT
- bib title: Effective test generation using pre-trained Large Language Models and
             mutation testing
- crossref title: **Effective test generation using pre-trained Large Language Models and mutation testing**
- doi: `10.1016/j.infsof.2024.107468`  · cited 6x
- registry venue: Information and Software Technology
- registry type: journal-article
- prose near \cite claims 'sbst', 'search-based' but real title is 'Effective test generation using pre-trained Large Language Models and mutation testing'
  - > …om docstrings or function signatures, and require no per-function search budget beyond inference time \citep{schafer2024,tufano2022}. Multiple recent studies have benchmarked LLM-generated tests against search-based baselines and reported competitive or superior effectiveness on standard benchmarks \citep{schafer2024,lemieux2023codamosa,dakhel2024mutation, straubinger2025debug,konstantinou2026newer}. The question that frames the present paper is therefore no longer ``can LLMs replace search-based generation?''--- the empirical answer to that is `…
  - > …d test quality and defect-detection capability are separate constructs, and that studies reporting only one of them are not interchangeable. \item \textbf{A per-operator LLM-vs-SBST comparison on matched Python functions.} Head-to-head LLM-vs-Pynguin comparison on mutation score is not itself new---\citet{dakhel2024mutation} and \citet{straubinger2025debug} both report one. What we add is the operator-level decomposition on a matched function set, which turns ``which para…
  - > …ion{Mutation-testing-based evaluation}\label{sec:related-mutation} Mutation testing has recently become the preferred effectiveness metric for LLM-generated tests, and the dominant use is \emph{generative} rather than evaluative: surviving mutants are fed back into the prompt to drive better tests. \citet{dakhel2024mutation} introduced this pattern with MuTAP, augmenting prompts with surviving mutants on Python benchmarks and reporting a 93.57\% mutation score against Pyn…

### `straubinger2025debug` — DESCRIPTION_SUSPECT
- bib title: Mutation Testing via Iterative Large Language Model-Driven
               Scientific Debugging
- crossref title: **Mutation Testing via Iterative Large Language Model-Driven Scientific Debugging**
- doi: `10.1109/ICSTW64639.2025.10962485`  · cited 6x
- registry venue: 2025 IEEE International Conference on Software Testing, Verification and Validation Workshops (ICSTW)
- registry type: proceedings-article
- prose near \cite claims 'sbst', 'search-based' but real title is 'Mutation Testing via Iterative Large Language Model-Driven Scientific Debugging'
  - > …om docstrings or function signatures, and require no per-function search budget beyond inference time \citep{schafer2024,tufano2022}. Multiple recent studies have benchmarked LLM-generated tests against search-based baselines and reported competitive or superior effectiveness on standard benchmarks \citep{schafer2024,lemieux2023codamosa,dakhel2024mutation, straubinger2025debug,konstantinou2026newer}. The question that frames the present paper is therefore no longer ``can LLMs replace search-based generation?''--- the empirical answer to that is `…
  - > …tion capability are separate constructs, and that studies reporting only one of them are not interchangeable. \item \textbf{A per-operator LLM-vs-SBST comparison on matched Python functions.} Head-to-head LLM-vs-Pynguin comparison on mutation score is not itself new---\citet{dakhel2024mutation} and \citet{straubinger2025debug} both report one. What we add is the operator-level decomposition on a matched function set, which turns ``which paradigm is better'' into ``which def…
  - > …pts with surviving mutants on Python benchmarks and reporting a 93.57\% mutation score against Pynguin and zero/few-shot baselines. \citet{wang2026mutgen} extend it with an iterative convergence mechanism on Java, \citet{bouafif2025primg} add a learned mutant-prioritisation module for Solidity, and \citet{straubinger2025debug} replace the feedback loop with simulated scientific debugging, in which the model forms and tests hypotheses about how to kill a specific mutant. At…

### `konstantinou2026newer` — DESCRIPTION_SUSPECT
- bib title: How well {LLM}-based test generation techniques perform with
               newer {LLM} versions?
- crossref title: **How well LLM-based test generation techniques perform with newer LLM versions?**
- doi: `10.1109/ICST69053.2026.00040`  · cited 5x
- registry venue: 2026 IEEE International Conference on Software Testing, Verification and Validation (ICST)
- registry type: proceedings-article
- prose near \cite claims 'mutation score', 'search-based' but real title is 'How well LLM-based test generation techniques perform with newer LLM versions?'
  - > …om docstrings or function signatures, and require no per-function search budget beyond inference time \citep{schafer2024,tufano2022}. Multiple recent studies have benchmarked LLM-generated tests against search-based baselines and reported competitive or superior effectiveness on standard benchmarks \citep{schafer2024,lemieux2023codamosa,dakhel2024mutation, straubinger2025debug,konstantinou2026newer}. The question that frames the present paper is therefore no longer ``can LLMs replace search-based generation?''--- the empirical answer to that is `…
  - > …nst runtime feedback, and provides the largest empirical comparison of frontier-LLM test-generation pipelines to date, covering coverage, fault detection, and runnable-test percentages across multiple LLMs and benchmarks. Several recent papers focus on specific LLM-test generation pipeline choices. \citet{konstantinou2026newer} replicate four such pipelines---HITS, SymPrompt, TestSpark and CoverUp---on 393 Java classes using current models, and find that a plain zero-shot pr…
  - > …Python & scientific-debugging loop & 1 & yes & yes \\ \citet{bouafif2025primg} & Solidity& mutant prioritisation & 1 & no & yes \\ \citet{wang2026mutgen} & Java & mutation feedback in prompt & 1 & yes & yes \\ \citet{wang2026mutationstudy} & Java & LLM \emph{mutant} generation& many & n/a & yes \\ \citet{konstantinou2026newer} & Java & 4 pipelines vs plain prompt & 3 & yes & yes \\ \citet{shin2026ragtest} & Python & RAG, 3 knowledge sources & 4 & yes & no \\ \citet{zhang202…

### `shin2026ragtest` — DESCRIPTION_SUSPECT
- bib title: Retrieval-Augmented Test Generation: How Far Are We?
- crossref title: **Retrieval-Augmented Test Generation: How Far Are We?**
- doi: `10.1145/3744916.3773163`  · cited 6x
- registry venue: Proceedings of the 2026 IEEE/ACM 48th International Conference on Software Engineering
- registry type: proceedings-article
- prose near \cite claims 'mutation score' but real title is 'Retrieval-Augmented Test Generation: How Far Are We?'
  - > …eparate the two. \end{itemize} \noindent Retrieval has been evaluated on adjacent code tasks such as repository-level completion \citep{zhang2023repocoder} and retrieval-augmented code generation \citep{su2025evor,lu2022reacc}. For \emph{unit test generation specifically} the literature is thinner: \citet{shin2026ragtest} and \citet{zhang2026refrag} are the two closest studies, and neither measures defect detection (\S\ref{sec:related-rag}). \noindent With the techniqu…
  - > …a human-evaluation component that operationalizes ``developer-perceived quality'' alongside the automated metrics. The present paper addresses all three gaps. \subsection{Retrieval-augmented generation for code}\label{sec:related-rag} Two recent studies apply retrieval directly to test generation. \citet{shin2026ragtest} compare basic-instruction prompting against two RAG configurations over three knowledge sources (API documentation, GitHub issues, Stack Overflow) fo…
  - > …of multiple RAG variants on code-completion benchmarks, finding that the best variant depends on the type of code-completion task. \paragraph{For RAG specifically applied to test generation,} the literature is thin but no longer empty, and two 2026 studies bound the question this paper sits inside. \citet{shin2026ragtest} compare basic-instruction prompting against two RAG configurations over three knowledge sources for five Python ML libraries across four LLMs, and fi…

## Description check (manual)

Metadata can match while the *prose* misdescribes the work — that is the
Huang & Huang failure. Read each context below against the real title.

### `watson2020`
- real title: **On learning meaningful assert statements for unit test cases**
  - > …977}. The \emph{ordinal} variant of $\alpha$, which we use, weights disagreements by how far apart the two ratings are on the scale. \subsection{LLM-based unit-test generation}\label{sec:related-llmtg} The use of large language models for unit-test synthesis predates the modern transformer-LLM era. \citet{watson2020} showed that sequence-to-sequence models could learn to generate assert statements from method bodies, evaluated against Java open-source projects. \c…

### `tufano2022`
- real title: **Methods2Test**
  - > …readable test code has shifted this landscape. LLMs trained on public source code can produce pytest- or JUnit-formatted test suites that read like hand-written tests, encode specifications drawn from docstrings or function signatures, and require no per-function search budget beyond inference time \citep{schafer2024,tufano2022}. Multiple recent studies have benchmarked LLM-generated tests against search-based baselines and reported competitive or superior effectiveness on st…
  - > …odels could learn to generate assert statements from method bodies, evaluated against Java open-source projects. \citet{tufano2022assert} scaled this approach with a BART-based encoder-decoder for assert generation, reporting improved accuracy over prior sequence models; the training data came from \citet{tufano2022}'s Methods2Test, a corpus of focal methods mapped to their test cases, which is a dataset contribution rather than a model. The arrival of decoder-onl…

### `lemieux2023codamosa`
- real title: **CodaMosa: Escaping Coverage Plateaus in Test Generation with Pre-trained Large Language Models**
  - > …om docstrings or function signatures, and require no per-function search budget beyond inference time \citep{schafer2024,tufano2022}. Multiple recent studies have benchmarked LLM-generated tests against search-based baselines and reported competitive or superior effectiveness on standard benchmarks \citep{schafer2024,lemieux2023codamosa,dakhel2024mutation, straubinger2025debug,konstantinou2026newer}. The question that frames the present paper is therefore no longer ``can LLMs replace search-based generation?''--- the empirical answer to that is `…
  - > …e models; the training data came from \citet{tufano2022}'s Methods2Test, a corpus of focal methods mapped to their test cases, which is a dataset contribution rather than a model. The arrival of decoder-only frontier LLMs (Codex, GPT-3.5, GPT-4) shifted research toward prompt-based test generation. \citet{lemieux2023codamosa} combined LLM prompting with search-based fallbacks, using the LLM to escape coverage plateaus where pure SBST runs got stuck. \citet{schafer2024} int…

### `schafer2024`
- real title: **An Empirical Evaluation of Using Large Language Models for Automated
             Unit Test Generation**
  - > …readable test code has shifted this landscape. LLMs trained on public source code can produce pytest- or JUnit-formatted test suites that read like hand-written tests, encode specifications drawn from docstrings or function signatures, and require no per-function search budget beyond inference time \citep{schafer2024,tufano2022}. Multiple recent studies have benchmarked LLM-generated tests against search-based baselines and reported competitive or superior effectiveness on st…
  - > …om docstrings or function signatures, and require no per-function search budget beyond inference time \citep{schafer2024,tufano2022}. Multiple recent studies have benchmarked LLM-generated tests against search-based baselines and reported competitive or superior effectiveness on standard benchmarks \citep{schafer2024,lemieux2023codamosa,dakhel2024mutation, straubinger2025debug,konstantinou2026newer}. The question that frames the present paper is therefore no longer ``can LLMs replace search-based generation?''--- the empirical answer to that is `…

### `siddiq2024junit`
- real title: **Using Large Language Models to Generate JUnit Tests: An Empirical Study**
  - > …t models, and find that a plain zero-shot prompt outperforms all four on line coverage, branch coverage and mutation score, at comparable query cost. \citet{schafer2024} evaluate TestPilot across three LLMs of differing capability and report that effectiveness tracks model size and training corpus. \citet{siddiq2024junit} evaluates the quality of code (including tests) generated by open-source code LLMs across multiple metrics. \citet{wang2025llm4se} provides a recent…
  - > …ort in \S\ref{sec:discussion-moe}, where qwen3-coder achieves a higher mutation kill rate than qwen3, but our annotators rate qwen3.5 higher on all three rubric dimensions. \paragraph{Human evaluation specifically for LLM-generated unit tests} is sparser than the broader code-generation literature. \citet{siddiq2024junit} reports an empirical evaluation of LLM-generated JUnit tests on multiple dimensions (compilability, correctness, coverage) but does not include a mul…

### `wang2025llm4se`
- real title: **Software Testing With Large Language Models: Survey, Landscape, and Vision**
  - > …rable query cost. \citet{schafer2024} evaluate TestPilot across three LLMs of differing capability and report that effectiveness tracks model size and training corpus. \citet{siddiq2024junit} evaluates the quality of code (including tests) generated by open-source code LLMs across multiple metrics. \citet{wang2025llm4se} provides a recent survey of LLM-for-software-testing work, mapping the rapid growth in this area from 2023 to 2025 and identifying open research dire…

### `lewis2020`
- real title: **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks**
  - > …d the dominant augmentation is retrieval. We introduce that idea first, then state the question this paper asks about it. \subsection{Retrieval augmentation for test generation}\label{sec:intro-rag} The augmentation we study is \emph{retrieval-augmented generation} (RAG). The original RAG framework \citep{lewis2020} augments an LLM's prompt with passages retrieved from a knowledge base; in the test-generation context, the knowledge base is typically a curated set…
  - > …mplementation, as the prior-paradigm reference. SBST tools derive assertions from \emph{observed} return values, which is the source of the \emph{regression oracle problem}: the assertions encode what the program currently does, not what it should do. \paragraph{Retrieval-augmented generation.} RAG \citep{lewis2020} prepends material retrieved from a corpus to a model's prompt. Retrieval here is \emph{dense}: both the query and each corpus chunk are embedded into…

### `parvez2021`
- real title: **Retrieval Augmented Code Generation and Summarization**
  - > …uld be read as evidence about that kind. The original RAG framework \citep{lewis2020} demonstrated that augmenting a sequence-generation LLM with a passage-retrieval step produces better outputs on knowledge-intensive NLP tasks. The framework has since been adapted to many code-generation contexts. \citet{parvez2021} showed that retrieval-augmented code summarization and generation could improve both code-completion and natural-language-to-code translation. \citet…

### `lu2022reacc`
- real title: **ReACC: A Retrieval-Augmented Code Completion Framework**
  - > …ph{relevance} from that of simply having more context in the prompt. Studies that omit this arm cannot separate the two. \end{itemize} \noindent Retrieval has been evaluated on adjacent code tasks such as repository-level completion \citep{zhang2023repocoder} and retrieval-augmented code generation \citep{su2025evor,lu2022reacc}. For \emph{unit test generation specifically} the literature is thinner: \citet{shin2026ragtest} and \citet{zhang2026refrag} are the two closest stud…
  - > …val step produces better outputs on knowledge-intensive NLP tasks. The framework has since been adapted to many code-generation contexts. \citet{parvez2021} showed that retrieval-augmented code summarization and generation could improve both code-completion and natural-language-to-code translation. \citet{lu2022reacc} demonstrated a retrieval-augmented code-completion framework using both lexical and semantic retrieval. \citet{zhang2023repocoder} introduced iterati…

### `zhang2023repocoder`
- real title: **RepoCoder: Repository-Level Code Completion Through Iterative Retrieval and Generation**
  - > …trast against Simple RAG isolates the contribution of retrieval \emph{relevance} from that of simply having more context in the prompt. Studies that omit this arm cannot separate the two. \end{itemize} \noindent Retrieval has been evaluated on adjacent code tasks such as repository-level completion \citep{zhang2023repocoder} and retrieval-augmented code generation \citep{su2025evor,lu2022reacc}. For \emph{unit test generation specifically} the literature is thinner: \cite…
  - > …ion contexts. \citet{parvez2021} showed that retrieval-augmented code summarization and generation could improve both code-completion and natural-language-to-code translation. \citet{lu2022reacc} demonstrated a retrieval-augmented code-completion framework using both lexical and semantic retrieval. \citet{zhang2023repocoder} introduced iterative retrieval at the repository level, where retrieval is re-run after each draft refinement —conceptually similar to our Iterative…

### `su2025evor`
- real title: **EvoR: Evolving Retrieval for Code Generation**
  - > …ph{relevance} from that of simply having more context in the prompt. Studies that omit this arm cannot separate the two. \end{itemize} \noindent Retrieval has been evaluated on adjacent code tasks such as repository-level completion \citep{zhang2023repocoder} and retrieval-augmented code generation \citep{su2025evor,lu2022reacc}. For \emph{unit test generation specifically} the literature is thinner: \citet{shin2026ragtest} and \citet{zhang2026refrag} are the two closest stud…
  - > …ced iterative retrieval at the repository level, where retrieval is re-run after each draft refinement —conceptually similar to our Iterative Critique baseline, though they evaluated on code completion rather than test generation. More recent work has explored variants of RAG specifically for code. \citet{su2025evor} introduces an evolving retrieval store that grows as code is generated, allowing later retrievals to benefit from earlier generations' decisions. \ci…

### `liu2025codereview`
- real title: **Refining ChatGPT-Generated Code: Characterizing and Mitigating Code Quality Issues**
  - > …gh they evaluated on code completion rather than test generation. More recent work has explored variants of RAG specifically for code. \citet{su2025evor} introduces an evolving retrieval store that grows as code is generated, allowing later retrievals to benefit from earlier generations' decisions. \citet{liu2025codereview} reports a head-to-head comparison of multiple RAG variants on code-completion benchmarks, finding that the best variant depends on the type of code-c…

### `rag4code2025`
- real title: **A Survey on Retrieval-Augmented Text Generation for Large Language Models**
  - > …original RAG framework \citep{lewis2020} augments an LLM's prompt with passages retrieved from a knowledge base; in the test-generation context, the knowledge base is typically a curated set of testing tutorials, framework documentation, and example test suites. The general framework is surveyed by \citet{rag4code2025}. Several variants have been applied to code tasks, and they differ in what they retrieve and how often: \begin{itemize} \item \emph{Simple RAG} perfo…

### `maynez2020`
- real title: **On Faithfulness and Factuality in Abstractive Summarization**
  - > …. \paragraph{Where our faithfulness result sits.} Our finding (\S\ref{sec:results-faithfulness}) that token-overlap faithfulness is a model-level trait, and that its apparent association with kill rate is confounded by model, connects to a broader literature on \emph{retrieval faithfulness} in NLP. \citet{maynez2020} and \citet{es2024ragas} propose faithfulness metrics for retrieval-augmented generation and discuss the gap between \emph{lexical} and \emph{semantic…

### `es2024ragas`
- real title: **RAGAs: Automated Evaluation of Retrieval Augmented Generation**
  - > …faithfulness result sits.} Our finding (\S\ref{sec:results-faithfulness}) that token-overlap faithfulness is a model-level trait, and that its apparent association with kill rate is confounded by model, connects to a broader literature on \emph{retrieval faithfulness} in NLP. \citet{maynez2020} and \citet{es2024ragas} propose faithfulness metrics for retrieval-augmented generation and discuss the gap between \emph{lexical} and \emph{semantic} faithfulness. Our resu…

### `demillo1978`
- real title: **Hints on Test Data Selection: Help for the Practicing Programmer**
  - > …ribution at their intersection. \subsection{Background and terminology}\label{sec:related-background} This subsection fixes the terms used throughout. Readers familiar with mutation testing and retrieval augmentation may skip to \S\ref{sec:related-llmtg}. \paragraph{Mutation testing.} Introduced by \citet{demillo1978}, mutation testing evaluates a test suite by injecting small, syntactically valid faults into the program under test and asking whether the suite noti…
  - > …We use rule-based AST operators rather than LLM-generated mutants for the same reason---\citet{wang2026mutationstudy} show the latter are more realistic, but they would introduce a second model-dependent factor into a design whose purpose is to isolate the first. Mutation testing was introduced by \citet{demillo1978} as a thought experiment about test-adequacy and was operationalized over the next 30 years into a workable empirical methodology. The foundational em…

### `andrews2005`
- real title: **Is mutation an appropriate tool for testing experiments?**
  - > …ion capability if it asserts only structural properties (return type, list length) rather than specific oracle values. The SE-relevant operationalization of ``do the tests catch bugs?'' is \emph{the mutation kill rate}—the fraction of systematically-injected code defects that the test suite detects \citep{andrews2005,just2014}. Mutation testing has been a gold-standard metric in the SBST literature for two decades but has been used only sporadically in LLM-test-generation e…
  - > …numeric boundary. Mutation testing is used here as an \emph{evaluation} metric; much recent work instead uses it \emph{generatively}, feeding surviving mutants back into the prompt (\S\ref{sec:related-mutation}). The metric earns its place by being a better proxy for fault detection than coverage. \citet{andrews2005} and \citet{just2014} showed that mutant detection correlates with real-fault detection more strongly than coverage does, which matters here because a…

### `just2014`
- real title: **Are mutants a valid substitute for real faults in software testing?**
  - > …ion capability if it asserts only structural properties (return type, list length) rather than specific oracle values. The SE-relevant operationalization of ``do the tests catch bugs?'' is \emph{the mutation kill rate}—the fraction of systematically-injected code defects that the test suite detects \citep{andrews2005,just2014}. Mutation testing has been a gold-standard metric in the SBST literature for two decades but has been used only sporadically in LLM-test-generation e…
  - > …ion testing is used here as an \emph{evaluation} metric; much recent work instead uses it \emph{generatively}, feeding surviving mutants back into the prompt (\S\ref{sec:related-mutation}). The metric earns its place by being a better proxy for fault detection than coverage. \citet{andrews2005} and \citet{just2014} showed that mutant detection correlates with real-fault detection more strongly than coverage does, which matters here because a suite can reach full…

### `coles2016`
- real title: **PIT: a practical mutation testing tool for Java (demo)**
  - > …tection rates of injected mutants are statistically correlated with detection rates of real faults from project bug-tracker history. \citet{just2014} provided a follow-up large-scale study on Java projects that confirmed the result. The mutation-testing tool ecosystem includes \textit{PIT} for Java \citep{coles2016} and \textit{mutmut} for Python. Our mutation operators (arithmetic, comparison, boundary, return-replacement, boolean-negation) are the canonical sub…

### `petrovic2018`
- real title: **State of mutation testing at google**
  - > …rovides the canonical recent survey of mutation testing, including the equivalent-mutant detection challenge that we address via ground-truth tests in \S\ref{sec:methods-mutation}, and the selective mutation strategies that motivate our per-operator decomposition in \S\ref{sec:results-peroperator}. \citet{petrovic2018} reports a large-scale industrial evaluation at Google showing that mutation testing remains a practically useful signal even at the scale of large pr…

### `papadakis2019survey`
- real title: **Mutation Testing Advances: An Analysis and Survey**
  - > …nt} when the injected change leaves observable behaviour unaltered---for instance replacing \verb|x < n| with \verb|x != n| inside a loop that increments \verb|x| by one---so that no test can kill it; equivalent mutants are excluded from the denominator, and detecting them is undecidable in general \citep{papadakis2019survey}. A \emph{mutation operator} is the rule that generates a class of mutants, such as replacing an arithmetic operator or perturbing a numeric boundary.…
  - > …rojects that confirmed the result. The mutation-testing tool ecosystem includes \textit{PIT} for Java \citep{coles2016} and \textit{mutmut} for Python. Our mutation operators (arithmetic, comparison, boundary, return-replacement, boolean-negation) are the canonical subset implemented by both tools. \citet{papadakis2019survey} provides the canonical recent survey of mutation testing, including the equivalent-mutant detection challenge that we address via ground-truth tests…

### `wang2026mutationstudy`
- real title: **A Comprehensive Study on Large Language Models for Mutation Testing**
  - > …model forms and tests hypotheses about how to kill a specific mutant. At industrial scale, \citet{foster2025ach} report Meta's ACH system generating privacy-hardening tests from LLM-produced mutants across 10{,}795 Kotlin classes. A complementary line asks how good LLM-generated \emph{mutants} are: \citet{wang2026mutationstudy} evaluate this on 851 real Java bugs and find LLM-generated mutants achieve 77.4\% real-bug detection against 41.6\% for rule-based operators. Our use…
  - > …surviving mutant ever enters a prompt in our pipeline; the mutation analysis runs strictly downstream of generation, which is what makes the kill rate an unbiased comparison across techniques that never saw it. We use rule-based AST operators rather than LLM-generated mutants for the same reason---\citet{wang2026mutationstudy} show the latter are more realistic, but they would introduce a second model-dependent factor into a design whose purpose is to isolate the first. Mut…

### `mcminn2004`
- real title: **Search‐based software test data generation: a survey**
  - > …suite can reach full statement coverage while asserting almost nothing. \paragraph{Search-based software testing (SBST).} The dominant pre-LLM paradigm treats test generation as an optimisation problem, evolving a population of candidate inputs against a coverage- or mutation-based fitness function \citep{mcminn2004,fraser2011}. We use Pynguin \citep{lukasczyk2022}, the standard Python implementation, as the prior-paradigm reference. SBST tools derive assertions from \emph{o…
  - > …families that later work builds on, motivating the kind of operator-level analysis we conduct in \S\ref{sec:results-peroperator}. \subsection{Search-based software testing}\label{sec:related-sbst} Search-based software testing has been the dominant paradigm for automated test-suite generation since \citet{mcminn2004}'s survey and \citet{harman2010}'s empirical comparison of search-based versus random testing. \emph{EvoSuite} \citep{fraser2011,fraser2013} is the ca…

### `harman2010`
- real title: **A Theoretical and Empirical Study of Search-Based Testing: Local, Global, and Hybrid Search**
  - > …on, motivating the kind of operator-level analysis we conduct in \S\ref{sec:results-peroperator}. \subsection{Search-based software testing}\label{sec:related-sbst} Search-based software testing has been the dominant paradigm for automated test-suite generation since \citet{mcminn2004}'s survey and \citet{harman2010}'s empirical comparison of search-based versus random testing. \emph{EvoSuite} \citep{fraser2011,fraser2013} is the canonical SBST tool for Java, comb…

### `fraser2011`
- real title: **EvoSuite**
  - > …2017}. The dominant paradigm prior to 2022 was \emph{search-based software testing} (SBST): tools like EvoSuite for Java and Pynguin for Python treat test-suite synthesis as an optimization problem, evolving a population of candidate test cases against a coverage- or mutation-based fitness function \citep{fraser2011,lukasczyk2022}. These tools achieve high branch coverage on self-contained functions and have demonstrated practical value in industrial deployments, but they suffe…
  - > …suite can reach full statement coverage while asserting almost nothing. \paragraph{Search-based software testing (SBST).} The dominant pre-LLM paradigm treats test generation as an optimisation problem, evolving a population of candidate inputs against a coverage- or mutation-based fitness function \citep{mcminn2004,fraser2011}. We use Pynguin \citep{lukasczyk2022}, the standard Python implementation, as the prior-paradigm reference. SBST tools derive assertions from \emph{o…

### `fraser2013`
- real title: **Whole Test Suite Generation**
  - > …\subsection{Search-based software testing}\label{sec:related-sbst} Search-based software testing has been the dominant paradigm for automated test-suite generation since \citet{mcminn2004}'s survey and \citet{harman2010}'s empirical comparison of search-based versus random testing. \emph{EvoSuite} \citep{fraser2011,fraser2013} is the canonical SBST tool for Java, combining genetic-algorithm test-case search with dynamic symbolic execution. EvoSuite has been validated repeat…

### `almasi2017`
- real title: **An Industrial Evaluation of Unit Test Generation: Finding Real Faults in a Financial Application**
  - > …\end{keyword} \end{frontmatter} \section{Introduction}\label{sec:introduction} Automated unit-test generation has been a target of empirical software-engineering research for decades, motivated by the well-documented cost of manual test authoring and the high marginal value of each additional test \citep{daka2014,almasi2017}. The dominant paradigm prior to 2022 was \emph{search-based software testing} (SBST): tools like EvoSuite for Java and Pynguin for Python treat test-…
  - > …t{harman2010}'s empirical comparison of search-based versus random testing. \emph{EvoSuite} \citep{fraser2011,fraser2013} is the canonical SBST tool for Java, combining genetic-algorithm test-case search with dynamic symbolic execution. EvoSuite has been validated repeatedly on industrial codebases \citep{almasi2017} and remains the reference baseline for Java-language SBST research. For Python, the corresponding tool is \emph{Pynguin} \citep{lukasczyk2023empirica…

### `lukasczyk2022`
- real title: **Pynguin: Automated Unit Test Generation for {P}ython**
  - > …2017}. The dominant paradigm prior to 2022 was \emph{search-based software testing} (SBST): tools like EvoSuite for Java and Pynguin for Python treat test-suite synthesis as an optimization problem, evolving a population of candidate test cases against a coverage- or mutation-based fitness function \citep{fraser2011,lukasczyk2022}. These tools achieve high branch coverage on self-contained functions and have demonstrated practical value in industrial deployments, but they suffe…
  - > …asserting almost nothing. \paragraph{Search-based software testing (SBST).} The dominant pre-LLM paradigm treats test generation as an optimisation problem, evolving a population of candidate inputs against a coverage- or mutation-based fitness function \citep{mcminn2004,fraser2011}. We use Pynguin \citep{lukasczyk2022}, the standard Python implementation, as the prior-paradigm reference. SBST tools derive assertions from \emph{observed} return values, which is the s…

### `lukasczyk2023empirical`
- real title: **An empirical study of automated unit test generation for Python**
  - > …BST tool for Java, combining genetic-algorithm test-case search with dynamic symbolic execution. EvoSuite has been validated repeatedly on industrial codebases \citep{almasi2017} and remains the reference baseline for Java-language SBST research. For Python, the corresponding tool is \emph{Pynguin} \citep{lukasczyk2023empirical,lukasczyk2022}. Pynguin combines coverage-driven genetic search with dynamic symbolic execution, optimized for Python's dynamic typing and runtime introspection cap…

### `smith1963`
- real title: **Retranslation of expectations: An approach to the construction of unambiguous anchors for rating scales.**
  - > …y reports the kind of three-annotator behaviorally anchored 0–5 rubric evaluation of RAG-augmented test generation that we present in \S\ref{sec:results-humaneval}. \paragraph{Rating-scale methodology.} The behaviorally-anchored rating scale (BARS) methodology we use in our rubric was introduced by \citet{smith1963} in industrial-psychology research and has been adapted for many software-engineering contexts since. \citet{landis1977} provides the canonical interp…
  - > …the same generated suites. To complement the automated mutation-testing analysis with a developer-perceived quality signal, we designed a three-dimension, behaviourally-anchored rating scale and ran a three-annotator study against it. The rubric is our own; the behaviourally-anchored format follows \citet{smith1963}, and we interpret agreement using the thresholds of \citet{landis1977} with ordinal Krippendorff's $\alpha$ \citep{krippendorff2018} as the primary t…

### `madeyski2024empirical`
- real title: **Empirical evaluation of continuous test-driven development in industrial settings**
  - > …lated-method} Our analytical methodology—mixed--effects regression with \texttt{sample\_idx} as a random intercept, Type-III ANOVA for unbalanced designs, Tukey HSD post-hoc, and Bonferroni correction across the family of pairwise tests—follows the standard recommendations of \citet{wohlin2012} and \citet{madeyski2024empirical} for analyzing empirical software-engineering experiments. The Spearman $\rho$ threshold of $\geq 0.8$ for cross-condition generalization is sourced f…

### `chen2021humaneval`
- real title: **Evaluating Large Language Models Trained on Code**
  - > …ing obvious garbage. Whether the above-threshold material is \emph{useful} is a separate question, and the Random RAG contrast (\S\ref{sec:results-killrate}) is what actually answers it. \subsection{Dataset}\label{sec:methods-dataset} We sampled 100 functions (seed = 42) from the union of HumanEval \citep{chen2021humaneval} and MBPP \citep{austin2021mbpp}, shuffled into a deterministic order. For the mutation-testing study, we ran each (technique $\times$ model) combinat…

### `austin2021mbpp`
- real title: **Program Synthesis with Large Language Models**
  - > …ove-threshold material is \emph{useful} is a separate question, and the Random RAG contrast (\S\ref{sec:results-killrate}) is what actually answers it. \subsection{Dataset}\label{sec:methods-dataset} We sampled 100 functions (seed = 42) from the union of HumanEval \citep{chen2021humaneval} and MBPP \citep{austin2021mbpp}, shuffled into a deterministic order. For the mutation-testing study, we ran each (technique $\times$ model) combination on all 100 samples, yielding…

### `daka2014`
- real title: **A Survey on Unit Testing Practices and Problems**
  - > …\end{keyword} \end{frontmatter} \section{Introduction}\label{sec:introduction} Automated unit-test generation has been a target of empirical software-engineering research for decades, motivated by the well-documented cost of manual test authoring and the high marginal value of each additional test \citep{daka2014,almasi2017}. The dominant paradigm prior to 2022 was \emph{search-based software testing} (SBST): tools like EvoSuite for Java and Pynguin for Python treat test-…

### `vaithilingam2022`
- real title: **Expectation vs. Experience: Evaluating the Usability of Code Generation Tools Powered by Large Language Models**
  - > …ll-rate comparison on matched Python functions, which is what we provide in \S\ref{sec:results-pynguin}. \subsection{Human evaluation of generated code and tests}\label{sec:related-humaneval} Human evaluation of LLM-generated code (and tests) is less mature than the literature on automated metrics. \citet{vaithilingam2022} at CHI established the foundational observation that developers' \emph{expectations} of LLM code-generation tools diverge sharply from their lived \e…
  - > …sess maintainability and understandability analytically, and \citet{foster2025ach} report engineer acceptance rates for an industrial deployment, but neither elicits blinded per-suite ratings from independent annotators. Prior human studies of LLM-generated code evaluate general Copilot suggestions \citep{vaithilingam2022,liang2024copilot} rather than the question of whether retrieval augmentation yields tests developers judge better. Pairing that signal with mutation testing on the sam…

### `liang2024copilot`
- real title: **A Large-Scale Survey on the Usability of AI Programming Assistants: Successes and Challenges**
  - > …} at CHI established the foundational observation that developers' \emph{expectations} of LLM code-generation tools diverge sharply from their lived \emph{experience}, finding that perceived usefulness depends heavily on readability and naming conventions even when objective correctness is similar. \citet{liang2024copilot} extended this observation to a large scale with a 410-developer survey of Copilot users at ICSE 2024, identifying readability and integration into th…
  - > …sess maintainability and understandability analytically, and \citet{foster2025ach} report engineer acceptance rates for an industrial deployment, but neither elicits blinded per-suite ratings from independent annotators. Prior human studies of LLM-generated code evaluate general Copilot suggestions \citep{vaithilingam2022,liang2024copilot} rather than the question of whether retrieval augmentation yields tests developers judge better. Pairing that signal with mutation testing on the sam…

### `mozannar2024chi`
- real title: **Reading Between the Lines: Modeling User Behavior and Costs in AI-Assisted Programming**
  - > …iang2024copilot} extended this observation to a large scale with a 410-developer survey of Copilot users at ICSE 2024, identifying readability and integration into the developer's existing workflow as the dominant quality dimensions—more important to participants than raw correctness in many cases. \citet{mozannar2024chi} at CHI 2024 complements these survey results with a behavioral-trace study of how developers actually invoke and edit Copilot's suggestions in practi…

### `landis1977`
- real title: **The Measurement of Observer Agreement for Categorical Data**
  - > …me artefacts we report Cohen's $\kappa$ for pairs and Krippendorff's $\alpha$ \citep{krippendorff2018} across all three, both of which correct observed agreement for the agreement expected by chance. Values run from $-1$ to $1$; $\alpha \geq 0.4$ is the conventional floor for ``moderate'' agreement \citep{landis1977}. The \emph{ordinal} variant of $\alpha$, which we use, weights disagreements by how far apart the two ratings are on the scale. \subsection{LLM-based…
  - > …we present in \S\ref{sec:results-humaneval}. \paragraph{Rating-scale methodology.} The behaviorally-anchored rating scale (BARS) methodology we use in our rubric was introduced by \citet{smith1963} in industrial-psychology research and has been adapted for many software-engineering contexts since. \citet{landis1977} provides the canonical interpretation thresholds for Cohen's $\kappa$ that we use to assess inter-rater agreement in \S\ref{sec:results-humaneval}: $…

### `wohlin2012`
- real title: **Experimentation in Software Engineering**
  - > …thodology}\label{sec:related-method} Our analytical methodology—mixed--effects regression with \texttt{sample\_idx} as a random intercept, Type-III ANOVA for unbalanced designs, Tukey HSD post-hoc, and Bonferroni correction across the family of pairwise tests—follows the standard recommendations of \citet{wohlin2012} and \citet{madeyski2024empirical} for analyzing empirical software-engineering experiments. The Spearman $\rho$ threshold of $\geq 0.8$ for cross-con…
  - > …area.} Two independent 2026 studies reach compatible conclusions by other routes (\S\ref{sec:related-gap}). The cumulative signal is only visible because those nulls were reported. \end{enumerate} \section{Threats to Validity}\label{sec:limitations} We organize the limitations along Wohlin et al.'s \citep{wohlin2012} standard taxonomy. \subsection{Construct validity}\label{sec:limitations-construct} \paragraph{Mutation kill rate is a proxy.} The five-operator set…

### `zar1984`
- real title: **Biostatistical Analysis**
  - > …erroni correction across the family of pairwise tests—follows the standard recommendations of \citet{wohlin2012} and \citet{madeyski2024empirical} for analyzing empirical software-engineering experiments. The Spearman $\rho$ threshold of $\geq 0.8$ for cross-condition generalization is sourced from \citet{zar1984} and is the threshold adopted by \citet{jureczko2015} for defect-prediction-model generalization across projects, which is the SE literature's nearest…

### `krippendorff2018`
- real title: **Content Analysis: An Introduction to Its Methodology**
  - > …how far the generated output reflects the retrieved material; we measure it both lexically and with a model judge (\S\ref{sec:methods-faithfulness}). \paragraph{Agreement statistics.} Where multiple annotators rate the same artefacts we report Cohen's $\kappa$ for pairs and Krippendorff's $\alpha$ \citep{krippendorff2018} across all three, both of which correct observed agreement for the agreement expected by chance. Values run from $-1$ to $1$; $\alpha \geq 0.4$ is th…
  - > …anonical interpretation thresholds for Cohen's $\kappa$ that we use to assess inter-rater agreement in \S\ref{sec:results-humaneval}: $\kappa < 0.20$ slight, $\kappa \in [0.21, 0.40]$ fair, $\kappa \in [0.41, 0.60]$ moderate, $\kappa \in [0.61, 0.80]$ substantial, $\kappa \geq 0.81$ almost perfect. \citet{krippendorff2018} defines the ordinal-$\alpha$ variant of inter-rater agreement we use as the primary three-rater statistic since Cohen's $\kappa$ is defined only pair…

### `jureczko2015`
- real title: **Cross--Project Defect Prediction With Respect To Code Ownership Model: An Empirical Study**
  - > …tests—follows the standard recommendations of \citet{wohlin2012} and \citet{madeyski2024empirical} for analyzing empirical software-engineering experiments. The Spearman $\rho$ threshold of $\geq 0.8$ for cross-condition generalization is sourced from \citet{zar1984} and is the threshold adopted by \citet{jureczko2015} for defect-prediction-model generalization across projects, which is the SE literature's nearest analog to our cross-LLM technique-ranking question.…

### `dakhel2024mutation`
- real title: **Effective test generation using pre-trained Large Language Models and mutation testing**
  - > …om docstrings or function signatures, and require no per-function search budget beyond inference time \citep{schafer2024,tufano2022}. Multiple recent studies have benchmarked LLM-generated tests against search-based baselines and reported competitive or superior effectiveness on standard benchmarks \citep{schafer2024,lemieux2023codamosa,dakhel2024mutation, straubinger2025debug,konstantinou2026newer}. The question that frames the present paper is therefore no longer ``can LLMs replace search-based generation?''--- the empirical answer to that is `…
  - > …d test quality and defect-detection capability are separate constructs, and that studies reporting only one of them are not interchangeable. \item \textbf{A per-operator LLM-vs-SBST comparison on matched Python functions.} Head-to-head LLM-vs-Pynguin comparison on mutation score is not itself new---\citet{dakhel2024mutation} and \citet{straubinger2025debug} both report one. What we add is the operator-level decomposition on a matched function set, which turns ``which para…

### `bouafif2025primg`
- real title: **PRIMG : Efficient LLM-driven Test Generation Using Mutant Prioritization**
  - > …tests. \citet{dakhel2024mutation} introduced this pattern with MuTAP, augmenting prompts with surviving mutants on Python benchmarks and reporting a 93.57\% mutation score against Pynguin and zero/few-shot baselines. \citet{wang2026mutgen} extend it with an iterative convergence mechanism on Java, \citet{bouafif2025primg} add a learned mutant-prioritisation module for Solidity, and \citet{straubinger2025debug} replace the feedback loop with simulated scientific debuggi…
  - > …4} & JS & doc-example prompt + repair & 3 & no & no \\ \citet{dakhel2024mutation} & Python & mutant-augmented prompt & 2 & yes & yes \\ \citet{foster2025ach} & Kotlin & LLM mutants + hardening & ind.\ & no & yes \\ \citet{straubinger2025debug} & Python & scientific-debugging loop & 1 & yes & yes \\ \citet{bouafif2025primg} & Solidity& mutant prioritisation & 1 & no & yes \\ \citet{wang2026mutgen} & Java & mutation feedback in prompt & 1 & yes & yes \\ \citet{wang2026mut…

### `foster2025ach`
- real title: **Mutation-Guided LLM-based Test Generation at Meta**
  - > …echanism on Java, \citet{bouafif2025primg} add a learned mutant-prioritisation module for Solidity, and \citet{straubinger2025debug} replace the feedback loop with simulated scientific debugging, in which the model forms and tests hypotheses about how to kill a specific mutant. At industrial scale, \citet{foster2025ach} report Meta's ACH system generating privacy-hardening tests from LLM-produced mutants across 10{,}795 Kotlin classes. A complementary line asks how g…
  - > …ootnotesize \setlength{\tabcolsep}{4pt} \begin{tabular}{@{}lllccc@{}} \toprule Study & Lang. & Technique studied & LLMs & Plain & Mut. \\ \midrule \citet{schafer2024} & JS & doc-example prompt + repair & 3 & no & no \\ \citet{dakhel2024mutation} & Python & mutant-augmented prompt & 2 & yes & yes \\ \citet{foster2025ach} & Kotlin & LLM mutants + hardening & ind.\ & no & yes \\ \citet{straubinger2025debug} & Python & scientific-debugging loop & 1 & yes & yes \\ \citet{…

### `straubinger2025debug`
- real title: **Mutation Testing via Iterative Large Language Model-Driven Scientific Debugging**
  - > …om docstrings or function signatures, and require no per-function search budget beyond inference time \citep{schafer2024,tufano2022}. Multiple recent studies have benchmarked LLM-generated tests against search-based baselines and reported competitive or superior effectiveness on standard benchmarks \citep{schafer2024,lemieux2023codamosa,dakhel2024mutation, straubinger2025debug,konstantinou2026newer}. The question that frames the present paper is therefore no longer ``can LLMs replace search-based generation?''--- the empirical answer to that is `…
  - > …tion capability are separate constructs, and that studies reporting only one of them are not interchangeable. \item \textbf{A per-operator LLM-vs-SBST comparison on matched Python functions.} Head-to-head LLM-vs-Pynguin comparison on mutation score is not itself new---\citet{dakhel2024mutation} and \citet{straubinger2025debug} both report one. What we add is the operator-level decomposition on a matched function set, which turns ``which paradigm is better'' into ``which def…

### `wang2026mutgen`
- real title: **Mutation-Guided Unit Test Generation With a Large Language Model**
  - > …er than evaluative: surviving mutants are fed back into the prompt to drive better tests. \citet{dakhel2024mutation} introduced this pattern with MuTAP, augmenting prompts with surviving mutants on Python benchmarks and reporting a 93.57\% mutation score against Pynguin and zero/few-shot baselines. \citet{wang2026mutgen} extend it with an iterative convergence mechanism on Java, \citet{bouafif2025primg} add a learned mutant-prioritisation module for Solidity, and \cit…
  - > …ion} & Python & mutant-augmented prompt & 2 & yes & yes \\ \citet{foster2025ach} & Kotlin & LLM mutants + hardening & ind.\ & no & yes \\ \citet{straubinger2025debug} & Python & scientific-debugging loop & 1 & yes & yes \\ \citet{bouafif2025primg} & Solidity& mutant prioritisation & 1 & no & yes \\ \citet{wang2026mutgen} & Java & mutation feedback in prompt & 1 & yes & yes \\ \citet{wang2026mutationstudy} & Java & LLM \emph{mutant} generation& many & n/a & yes \\ \cit…

### `konstantinou2026newer`
- real title: **How well LLM-based test generation techniques perform with newer LLM versions?**
  - > …om docstrings or function signatures, and require no per-function search budget beyond inference time \citep{schafer2024,tufano2022}. Multiple recent studies have benchmarked LLM-generated tests against search-based baselines and reported competitive or superior effectiveness on standard benchmarks \citep{schafer2024,lemieux2023codamosa,dakhel2024mutation, straubinger2025debug,konstantinou2026newer}. The question that frames the present paper is therefore no longer ``can LLMs replace search-based generation?''--- the empirical answer to that is `…
  - > …nst runtime feedback, and provides the largest empirical comparison of frontier-LLM test-generation pipelines to date, covering coverage, fault detection, and runnable-test percentages across multiple LLMs and benchmarks. Several recent papers focus on specific LLM-test generation pipeline choices. \citet{konstantinou2026newer} replicate four such pipelines---HITS, SymPrompt, TestSpark and CoverUp---on 393 Java classes using current models, and find that a plain zero-shot pr…

### `zhang2026refrag`
- real title: **Reference-Based Retrieval-Augmented Unit Test Generation**
  - > …e} \noindent Retrieval has been evaluated on adjacent code tasks such as repository-level completion \citep{zhang2023repocoder} and retrieval-augmented code generation \citep{su2025evor,lu2022reacc}. For \emph{unit test generation specifically} the literature is thinner: \citet{shin2026ragtest} and \citet{zhang2026refrag} are the two closest studies, and neither measures defect detection (\S\ref{sec:related-rag}). \noindent With the technique space in view, the questio…
  - > …test} compare basic-instruction prompting against two RAG configurations over three knowledge sources (API documentation, GitHub issues, Stack Overflow) for five Python ML libraries across four LLMs, and report that retrieval does not improve correctness while adding 6.5\% line coverage on average. \citet{zhang2026refrag} take a different route: rather than retrieving documentation, their RefTest retrieves the \emph{existing tests of related methods}, decomposing the r…

### `shin2026ragtest`
- real title: **Retrieval-Augmented Test Generation: How Far Are We?**
  - > …eparate the two. \end{itemize} \noindent Retrieval has been evaluated on adjacent code tasks such as repository-level completion \citep{zhang2023repocoder} and retrieval-augmented code generation \citep{su2025evor,lu2022reacc}. For \emph{unit test generation specifically} the literature is thinner: \citet{shin2026ragtest} and \citet{zhang2026refrag} are the two closest studies, and neither measures defect detection (\S\ref{sec:related-rag}). \noindent With the techniqu…
  - > …a human-evaluation component that operationalizes ``developer-perceived quality'' alongside the automated metrics. The present paper addresses all three gaps. \subsection{Retrieval-augmented generation for code}\label{sec:related-rag} Two recent studies apply retrieval directly to test generation. \citet{shin2026ragtest} compare basic-instruction prompting against two RAG configurations over three knowledge sources (API documentation, GitHub issues, Stack Overflow) fo…

### `tufano2022assert`
- real title: **Generating accurate assert statements for unit test cases using pretrained transformers**
  - > …eneration}\label{sec:related-llmtg} The use of large language models for unit-test synthesis predates the modern transformer-LLM era. \citet{watson2020} showed that sequence-to-sequence models could learn to generate assert statements from method bodies, evaluated against Java open-source projects. \citet{tufano2022assert} scaled this approach with a BART-based encoder-decoder for assert generation, reporting improved accuracy over prior sequence models; the training da…
