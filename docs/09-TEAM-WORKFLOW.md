# 09 — Team Workflow & Division of Labor

---

## 1. Role Definitions

### Person A — Pipeline & Application Lead
Primary ownership:
- Whisper transcription pipeline (`src/transcribe.py`)
- Chunking logic (`src/chunking.py`)
- Streamlit application (`app.py`)
- Results visualization notebooks
- EDA notebook
- README and documentation

### Person B — Models & Evaluation Lead
Primary ownership:
- Preprocessing pipeline (`src/preprocess.py`)
- TextRank baseline (`src/summarize_extractive.py`)
- BART / T5 integration (`src/summarize_abstractive.py`)
- Evaluation framework (`src/evaluate.py`)
- Ablation experiments
- Error analysis
- Unit tests

### Shared Responsibilities
- Technical report writing (split by section)
- Presentation creation and rehearsal
- Code review (review each other's PRs)
- Dataset preparation
- Final integration testing

---

## 2. Git Workflow

### Branch Strategy

```
main
  │
  ├── feature/whisper-pipeline        (Person A)
  ├── feature/preprocessing           (Person B)
  ├── feature/textrank-baseline       (Person B)
  ├── feature/bart-integration        (Person B)
  ├── feature/t5-integration          (Person A)
  ├── feature/chunking                (Person A)
  ├── feature/evaluation              (Person B)
  ├── feature/streamlit-app           (Person A)
  ├── feature/ablation-experiments    (Both)
  └── docs/report                     (Both)
```

### Rules
1. Never push directly to `main`
2. Create a feature branch for each module
3. Merge to `main` once the feature is tested and working
4. Write meaningful commit messages: `"Add TextRank summarizer with sumy integration"`
5. Keep commits atomic — one logical change per commit

### Commit Message Format

```
<type>: <short description>

Types:
  feat:     New feature or module
  fix:      Bug fix
  refactor: Code restructuring without changing behavior
  test:     Adding or updating tests
  docs:     Documentation changes
  data:     Data pipeline or dataset changes
  eval:     Evaluation-related changes
  exp:      Experiment or ablation
  app:      Streamlit application changes

Examples:
  feat: add Whisper batch transcription with progress bar
  fix: handle empty transcript in TextRank summarizer
  eval: add BERTScore computation to evaluate.py
  exp: run ablation A3 — Whisper model size comparison
  app: add summary comparison tab to Streamlit UI
```

---

## 3. Communication Plan

### Meetings
- **Daily standup** (10 min, async via chat): What I did, what I'll do, any blockers
- **Weekly sync** (30 min, video call): Review progress, plan next week, discuss problems
- **Ad-hoc pairing**: For integration work or debugging complex issues

### Tools
- **Chat**: Slack / Discord / iMessage (whatever you normally use)
- **Code**: GitHub (or local git if no GitHub)
- **Shared docs**: Google Drive for report drafts
- **Compute**: Shared Colab notebooks for GPU experiments

---

## 4. Master Timeline (All Phases)

```
Week 1: Mar 31 - Apr 6    PHASE 2 — Data & Baseline
────────────────────────────────────────────────────
Day 1 (Mon 3/31)  ┃ Both  ┃ Repo setup, deps, .gitignore
Day 2 (Tue 4/1)   ┃ A     ┃ Download AMI, explore data
                   ┃ B     ┃ Implement preprocess.py + tests
Day 3 (Wed 4/2)   ┃ A     ┃ Implement transcribe.py
                   ┃ B     ┃ Implement summarize_extractive.py
Day 4 (Thu 4/3)   ┃ A     ┃ Run Whisper on test set (Colab)
                   ┃ B     ┃ Implement evaluate.py
Day 5 (Fri 4/4)   ┃ Both  ┃ Run baseline experiments
Day 6 (Sat 4/5)   ┃ A     ┃ EDA notebook
                   ┃ B     ┃ Error analysis on baseline
Day 7 (Sun 4/6)   ┃ Both  ┃ Write progress report
Day 8 (Mon 4/7)   ┃ Both  ┃ Submit Phase 2

Week 2: Apr 8 - Apr 13    PHASE 3 — Models (Part 1)
────────────────────────────────────────────────────
Day 1 (Tue 4/8)   ┃ A     ┃ Implement chunking.py
                   ┃ B     ┃ Start AbstractiveSummarizer class
Day 2 (Wed 4/9)   ┃ A     ┃ T5 integration
                   ┃ B     ┃ BART integration
Day 3 (Thu 4/10)  ┃ A     ┃ Run T5 on test set (Colab)
                   ┃ B     ┃ Run BART on test set (Colab)
Day 4 (Fri 4/11)  ┃ Both  ┃ Hierarchical summarization
Day 5 (Sat 4/12)  ┃ A     ┃ Ablation A1 (preprocessing)
                   ┃ B     ┃ Ablation A2 (model comparison)
Day 6 (Sun 4/13)  ┃ A     ┃ Ablation A3 (Whisper size)
                   ┃ B     ┃ Ablation A4 (chunk size)

Week 3: Apr 14 - Apr 21   PHASE 3 — Models (Part 2)
────────────────────────────────────────────────────
Day 1 (Mon 4/14)  ┃ Both  ┃ Ablations A5-A7
Day 2 (Tue 4/15)  ┃ A     ┃ Compile results tables + plots
                   ┃ B     ┃ Error analysis (20 samples per model)
Day 3 (Wed 4/16)  ┃ A     ┃ Results visualization notebook
                   ┃ B     ┃ Write error analysis section
Day 4 (Thu 4/17)  ┃ Both  ┃ Hyperparameter tuning
Day 5 (Fri 4/18)  ┃ A     ┃ Start Streamlit app skeleton
                   ┃ B     ┃ Statistical significance tests
Day 6 (Sat 4/19)  ┃ Both  ┃ Integration testing
Day 7 (Sun 4/20)  ┃ Both  ┃ Write Phase 3 progress report
Day 8 (Mon 4/21)  ┃ Both  ┃ Submit Phase 3

Week 4: Apr 22 - Apr 28   PHASE 4 — Polish (Part 1)
────────────────────────────────────────────────────
Day 1-3 (4/22-24) ┃ A     ┃ Complete Streamlit app
                   ┃ B     ┃ Report: Intro + Related Work
Day 4-5 (4/25-26) ┃ A     ┃ Enhancement E3 (action items)
                   ┃ B     ┃ Report: Methodology
Day 6-7 (4/27-28) ┃ Both  ┃ Final experiments, fill result tables

Week 5: Apr 29 - May 5    PHASE 4 — Polish (Part 2)
────────────────────────────────────────────────────
Day 1-2 (4/29-30) ┃ Both  ┃ Report: Experiments + Analysis
Day 3 (5/1)       ┃ Both  ┃ Human evaluation (3 raters × 20 samples)
Day 4-5 (5/2-3)   ┃ A     ┃ Unit tests, code cleanup
                   ┃ B     ┃ Report: Discussion + Conclusion
Day 6-7 (5/4-5)   ┃ Both  ┃ Report: Abstract, References, formatting

Week 6: May 6 - May 13    PHASE 4 — Final
────────────────────────────────────────────────────
Day 1 (5/6)       ┃ Both  ┃ Create presentation slides
Day 2 (5/7)       ┃ Both  ┃ Rehearse presentation
Day 3 (5/8)       ┃ A     ┃ Prepare demo, test on presentation machine
Day 4 (5/9)       ┃ Both  ┃ Final report peer-review + edit
Day 5 (5/10)      ┃ Both  ┃ BUFFER — fix anything outstanding
Day 6 (5/11)      ┃ Both  ┃ Final README, .gitignore, repo cleanup
Day 7 (5/12)      ┃ Both  ┃ Submit report + code
Day 8 (5/13)      ┃ Both  ┃ *** PRESENTATION DAY ***
```

---

## 5. Definition of Done (per module)

A module is "done" when:
- [ ] Code runs without errors
- [ ] Function docstrings are complete
- [ ] Type hints on all signatures
- [ ] At least 2 unit tests per public function
- [ ] Tested on at least 3 real data samples
- [ ] Config values loaded from `config.yaml` (no hardcoded magic numbers)
- [ ] Reviewed by the other team member
- [ ] Merged to `main`

---

## 6. Grading Alignment Checklist

Map deliverables to rubric categories to ensure nothing is missed:

### Technical Quality (40 pts)
- [ ] **Implementation (20 pts)**: Clean code, correct, efficient
  - All `src/` modules complete and tested
  - Config-driven, no hardcoded values
  - Proper error handling
- [ ] **Methodology (10 pts)**: Sound experimental design
  - Justified model choices
  - Proper train/test splits
  - Reproducible with random seeds
- [ ] **Innovation (10 pts)**: Creative solutions
  - Hierarchical chunking for long transcripts
  - Multiple model comparison
  - Enhancement features (action items, configurable length)

### Evaluation & Analysis (25 pts)
- [ ] **Experimental rigor (15 pts)**: Baselines, metrics, statistics
  - TextRank baseline established
  - ROUGE + BERTScore + human eval
  - Statistical significance tests
  - 7+ ablation experiments
- [ ] **Error analysis (10 pts)**: Understanding failures
  - Error taxonomy defined
  - 3-5 concrete failure examples
  - Per-model error distribution

### Communication (25 pts)
- [ ] **Written report (15 pts)**: Clear, organized, accurate
  - 15-20 pages, proper structure
  - 15+ references
  - Figures and tables
- [ ] **Oral presentation (10 pts)**: Clear delivery, effective visuals
  - 15 min + 5 min Q&A
  - Live demo
  - Rehearsed

### Collaboration (10 pts)
- [ ] **Team contribution (5 pts)**: Equal participation
  - Git commit history shows balanced contributions
  - Peer evaluation form
- [ ] **Project management (5 pts)**: Meeting deadlines
  - All 4 phase deliverables submitted on time
  - Documentation maintained throughout
