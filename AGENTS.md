# AGENTS.md

## Goal
Refactor a Fashion-MNIST HOG/PCA/SVM project into a portfolio-ready inference pipeline and similarity search system.

## Priorities
1. inference pipeline
2. HOG/PCA-based embedding storage
3. similarity search
4. Optuna + StratifiedKFold validation
5. FastAPI (optional, last)

## Constraints
- Keep SVM as the core classifier
- Do not introduce deep learning models
- Do not use LLM, LangChain, or vector DB
- Junior-level scope only

## Rules
- Reuse existing notebook logic
- Do not over-engineer
- Keep changes minimal and modular
- Work step-by-step (one feature at a time)

## Validation
- inference must run end-to-end on a sample image
- similarity search must return top-k results
- Optuna must use StratifiedKFold

## Data Rules
- Do not hardcode absolute dataset paths
- Use environment variables (via .env file) for dataset location
- Assume datasets are stored outside the repository

## Code Style Rules
- Use Python 3.11+ syntax
- Prefer small functions with single responsibility
- Use `X | None` instead of `Optional[X]`
- Avoid overly long functions; if a function grows past ~40 lines, consider splitting it
- Avoid unnecessary inline comments
- Add short docstrings only for public functions
- Reuse notebook logic instead of rewriting from scratch
- Refactor notebook code into clean functions instead of copying entire cells
- Keep file structure simplse and minimal

## Data & Path Rules
- Do not hardcode absolute file paths
- Use environment variables (.env) for dataset paths

## Implementation Rules
- Functions should be composable (preprocessing → HOG → PCA → predict)
- Inference functions must have a clear input/output interface
- Separate training and inference logic
- Do not re-fit PCA or SVM during inference; use pre-trained models
- Do not leave debug print statements in final code
- Each file should have a single responsibility

## Commit Rules
- Use prefix: feat, refactor, fix, chore, style
- Write commit messages in Korean

## Comment Rules
- Use Korean docstrings for functions
- Add at most 1–3 short Korean comments inside functions to explain key steps
- Do not repeat the docstring in comments
- Avoid obvious or line-by-line comments
