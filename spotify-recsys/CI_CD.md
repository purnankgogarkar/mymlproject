# GitHub Actions CI/CD Pipeline

## Overview

This project uses GitHub Actions for continuous integration and deployment. The CI pipeline automatically runs on every push to `main` and on pull requests.

## Workflow File

**Location:** `.github/workflows/ci.yml`

The workflow consists of 5 jobs:

### Job 1: Test
**Purpose:** Run automated tests on every commit

**Details:**
- Runs on: `ubuntu-latest` with Python 3.9
- Steps:
  1. Checkout code
  2. Set up Python environment
  3. Cache pip dependencies
  4. Install requirements
  5. Run pytest with verbose output
  6. Upload test artifacts

**Triggers on:**
- Push to `main` or `develop`
- Pull requests to `main` or `develop`

**Failure criteria:**
- Any test fails (Exit code ≠ 0)

### Job 2: Lint
**Purpose:** Check code quality and style

**Details:**
- Runs on: `ubuntu-latest`
- Linters used:
  - **flake8**: PEP 8 compliance, complexity checks
  - **ruff**: Fast Python linter with extended checks
  - **black**: Code formatting check
  - **isort**: Import sorting check

**Checks:**
```bash
flake8 src/ app/          # Style & errors
ruff check src/ app/      # Extended checks
black --check src/ app/   # Formatting
isort --check-only src/ app/  # Import order
```

**Configuration files:**
- `.flake8` — Flake8 settings
- `pyproject.toml` — Ruff, Black, isort settings

**Failure criteria:**
- Set to `continue-on-error: true` (warnings only, doesn't block)

### Job 3: Security
**Purpose:** Detect security vulnerabilities

**Details:**
- Uses:
  - **bandit**: Security vulnerability scanner
  - **safety**: Dependency vulnerability checker

**Checks:**
```bash
bandit -r src/ app/      # Code security scan
safety check             # Dependency audit
```

**Failure criteria:**
- Set to `continue-on-error: true` (warnings only)

### Job 4: Build
**Purpose:** Build Docker image on main branch

**Details:**
- Runs only if:
  - Tests pass ✓
  - Lint passes ✓
  - Main branch push ✓
- Uses Docker Buildx for efficient builds
- Caches layers for faster builds
- Tags with commit SHA

**Triggers:** Push to `main` after Test & Lint pass

### Job 5: Notify
**Purpose:** Report overall pipeline status

**Details:**
- Runs after all other jobs
- Aggregates results from Test, Lint, Security
- Generates GitHub summary
- Fails if any critical job failed

---

## Local Development

### Run tests locally
```bash
python -m pytest tests/ -v
```

### Check linting
```bash
# Flake8
flake8 src/ app/

# Ruff
ruff check src/ app/

# Black
black src/ app/

# isort
isort src/ app/
```

### Fix linting issues automatically
```bash
# Black (format)
black src/ app/

# isort (sort imports)
isort src/ app/

# Ruff (auto-fix some issues)
ruff check --fix src/ app/
```

### Run security scan
```bash
bandit -r src/ app/
safety check
```

---

## Configuration Files

### `.flake8`
PEP 8 style checking:
- Line length: 120 chars
- Ignores: E203, E501, W503, W504
- Complexity limit: 10

### `pyproject.toml`
Multi-tool configuration:
- **Black**: Line length 120, Python 3.9+
- **isort**: Black-compatible profile
- **Ruff**: Extended checks for E, W, F, I, C, B

### `pytest.ini`
Test runner configuration:
- Discovery patterns
- Verbose output
- Strict markers
- Short traceback format

---

## Workflow Triggers

### On Push to Main
All 5 jobs run:
```
push: [main] → Test → Lint → Security → Build → Notify
```

### On Pull Request to Main
First 3 jobs run (no build):
```
pull_request: [main] → Test → Lint → Security → Notify
```

### Branch Restrictions
- Tests: `main`, `develop`
- Linting: `main`, `develop`
- Building: `main` only (on push)

---

## Artifacts

### Generated on Success
- Test results cached
- Coverage reports (if enabled)
- Docker image built (on main)

### Uploaded Artifacts
```
pytest-results/
  .pytest_cache/
  htmlcov/
```

---

## Status Checks

### Required Checks (for main branch)
Recommend enabling in GitHub settings:
- ✅ `Test` must pass
- ✅ `Lint` (recommended)
- ✅ `Security` (recommended)

### How to Require Checks
1. Go to repo → Settings → Branches
2. Select `main` branch
3. Add required status checks:
   - `Test`
   - `Lint`
   - `Security`
4. Enable "Require status checks to pass before merging"

---

## Troubleshooting

### Tests fail in CI but pass locally
**Possible causes:**
- Different Python version
- Missing dependencies in requirements.txt
- Path issues (use relative paths)

**Solution:**
```bash
# Match CI environment
python3.9 -m pytest tests/ -v
```

### Lint warnings block PR
**Solution:**
- Fix formatting: `black src/ app/`
- Fix imports: `isort src/ app/`
- Check flake8: `flake8 src/ app/`

### Build fails after tests pass
**Check:**
- `Dockerfile` exists
- All required files included
- No circular dependencies in imports

### Cache not working
**Clear cache:**
- GitHub Actions → All Workflows → ⋮ → Clear all caches

---

## Performance Tips

### Speed up CI/CD
1. **Caching:** Dependencies cached (~90% faster on repeat)
2. **Parallel jobs:** Test and Lint run simultaneously
3. **Conditional steps:** Build only on main branch
4. **Shallow clone:** `actions/checkout@v3` uses shallow clones

### Current typical times
- Test job: 1-2 minutes
- Lint job: 30-60 seconds
- Security job: 30-60 seconds
- Total: 2-3 minutes

---

## Monitoring

### View workflow runs
1. Go to repo → Actions
2. Select workflow: `CI Pipeline`
3. View job logs and artifacts

### View recent runs
```
https://github.com/YOUR_ORG/spotify-recsys/actions
```

### Annotations in PR
- ✅ Pass: Green checkmark
- ❌ Fail: Red X with error details
- ⚠️  Warning: Yellow warning in annotation

---

## Next Steps

1. **Push to main branch** to trigger CI
2. **Monitor** the workflow run in Actions tab
3. **Fix** any failures shown
4. **Configure** branch protection rules
5. **Enable** status checks requirement

---

## Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Pytest Guide](https://docs.pytest.org/)
- [Flake8 Documentation](https://flake8.pycqa.org/)
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [Black Code Formatter](https://black.readthedocs.io/)
