# Plan 0 — Rename `airproxy` → `ai-engine`

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename the entire project from `airproxy` to `ai-engine` in one atomic, verifiable pass — without breaking any v0.1 functionality.

**Architecture:** Single feature branch off `feat/gateway-core` (or its successor). Mechanical rename of crate names, identifiers, binary, config paths, docs. Validation gate: `cargo test --workspace` produces the same pass count before and after; clippy clean; the binary's `--check` smoke test still works.

**Tech Stack:** No new tech. `sed`, `git mv`, `find`, `cargo`. No code changes other than identifier renames.

**Scope rule:** This plan ONLY renames. It does NOT add features, refactor, or "clean up" anything found along the way. Any deviation from a pure rename is out of scope and gets a TODO note instead.

---

## File-by-file rename inventory (pre-task survey)

Files / directories that need rename or content change:

**Directory renames:**
- `crates/airproxy/` → `crates/ai-engine/`
- `crates/airproxy-core/` → `crates/ai-engine-core/`
- `crates/airproxy-provider/` → `crates/ai-engine-provider/`
- `crates/airproxy-openai/` → `crates/ai-engine-openai/`
- `crates/airproxy-anthropic/` → `crates/ai-engine-anthropic/`
- `crates/airproxy-stages/` → `crates/ai-engine-stages/`
- `crates/airproxy-config/` → `crates/ai-engine-config/`
- `crates/airproxy-http/` → `crates/ai-engine-http/`

**File renames:**
- `airproxy.toml.example` → `ai-engine.toml.example`

**File content changes (identifier substitution):**
- Root `Cargo.toml` — workspace.dependencies path entries (`airproxy-*` → `ai-engine-*`)
- Every crate's `Cargo.toml` — `[package].name` field + internal dep references
- Every `.rs` source file — `use airproxy_*::*` imports, doc comments referencing the old name, log targets
- `README.md` — every reference to airproxy
- `crates/ai-engine/src/cli.rs` — clap binary name + about text
- `crates/ai-engine/src/lib.rs` — module doc
- `crates/ai-engine/src/main.rs` — argv 0 references, if any
- `ai-engine.toml.example` — comment headers that mention the project name; default config file paths in examples
- `LICENSE` — copyright attribution line (already says "ai-engine contributors", confirmed in this branch)
- `NOTICE` — already says ai-engine (confirmed)
- `docs/superpowers/specs/2026-05-23-airproxy-gateway-core-design.md` — leave as historical document but add a header note that the project was renamed (optional; we can also leave it untouched as a snapshot).
- `docs/superpowers/plans/2026-05-23-airproxy-gateway-core.md` — same: historical document.
- `.gitignore` — should not need changes (it ignores `/target`, etc., no project-name references).

**Things that explicitly do NOT change:**
- Crate version (still `0.1.0`).
- Test bodies (no behavioral assertions reference the project name).
- The repo directory itself at `/home/alessio/aip/airproxy/` — we leave the *outer* directory named `airproxy/` for now; renaming the working-directory path is the operator's choice (some tooling caches paths), and is independent of the in-tree rename. We note this in the README under "moving to a renamed checkout" if the operator wants to do it later.
- The git remote URL — none configured currently; left untouched.

---

## Working assumptions

- Branch: tasks happen on a new branch `chore/rename-to-ai-engine` cut from `feat/gateway-core` HEAD.
- Working directory: `/home/alessio/aip/airproxy/` (the outer dir name does not change as part of this plan).
- The user prefers commits **without** Co-Authored-By footers (global preference from CLAUDE.md).
- All `cargo` invocations must check exit codes properly — do not pipe to `tail` and rely on the pipeline exit code, which masks failures. Run `cargo` directly first, then optionally pipe for display.

---

### Task 1: Create rename branch and capture baseline

**Files:** none (git operations only)

- [ ] **Step 1: Verify current branch state**

```bash
cd /home/alessio/aip/airproxy
git status                      # expected: clean
git log --oneline -1            # expected: v0.1.0 tagged commit or close
git tag                         # expected: v0.1.0 present
```

- [ ] **Step 2: Create rename branch off current HEAD**

```bash
git checkout -b chore/rename-to-ai-engine
git branch --show-current       # expected: chore/rename-to-ai-engine
```

- [ ] **Step 3: Capture baseline test count and clippy state**

```bash
cargo test --workspace 2>&1 | tee /tmp/before-rename-tests.txt
grep -E "^test result:" /tmp/before-rename-tests.txt
cargo clippy --workspace --all-targets -- -D warnings 2>&1 | tee /tmp/before-rename-clippy.txt
echo $?                         # expected: 0
```

Record the number of passing tests (should be 78 + ~3 wire-compat + 1 load smoke ignored = ~82). This is the post-rename verification target.

- [ ] **Step 4: No commit yet**

Branch is in place; baseline captured. We do not commit anything in Task 1.

---

### Task 2: Rename crate directories on disk

**Files:**
- Move: `crates/airproxy/` → `crates/ai-engine/`
- Move: `crates/airproxy-core/` → `crates/ai-engine-core/`
- Move: `crates/airproxy-provider/` → `crates/ai-engine-provider/`
- Move: `crates/airproxy-openai/` → `crates/ai-engine-openai/`
- Move: `crates/airproxy-anthropic/` → `crates/ai-engine-anthropic/`
- Move: `crates/airproxy-stages/` → `crates/ai-engine-stages/`
- Move: `crates/airproxy-config/` → `crates/ai-engine-config/`
- Move: `crates/airproxy-http/` → `crates/ai-engine-http/`

- [ ] **Step 1: `git mv` each crate directory**

```bash
cd /home/alessio/aip/airproxy
git mv crates/airproxy           crates/ai-engine
git mv crates/airproxy-core      crates/ai-engine-core
git mv crates/airproxy-provider  crates/ai-engine-provider
git mv crates/airproxy-openai    crates/ai-engine-openai
git mv crates/airproxy-anthropic crates/ai-engine-anthropic
git mv crates/airproxy-stages    crates/ai-engine-stages
git mv crates/airproxy-config    crates/ai-engine-config
git mv crates/airproxy-http      crates/ai-engine-http
```

- [ ] **Step 2: Verify directory listing**

```bash
ls crates/
# Expected output (alphabetical):
# ai-engine  ai-engine-anthropic  ai-engine-config  ai-engine-core  ai-engine-http  ai-engine-openai  ai-engine-provider  ai-engine-stages
```

- [ ] **Step 3: Confirm cargo cannot yet build**

```bash
cargo check --workspace 2>&1 | tail -20
# Expected: errors. Workspace path references in root Cargo.toml still point to airproxy-* paths.
# This is intentional — we fix it in Task 3.
```

- [ ] **Step 4: No commit yet** — the workspace is broken in this state; commits land at the end of Task 4 once cargo builds again.

---

### Task 3: Update workspace `Cargo.toml` path references

**Files:**
- Modify: `Cargo.toml` (workspace root)

- [ ] **Step 1: Edit workspace dependency paths**

Replace the internal-crate path entries in `[workspace.dependencies]`:

```toml
# OLD
airproxy-core      = { path = "crates/airproxy-core" }
airproxy-provider  = { path = "crates/airproxy-provider" }
airproxy-openai    = { path = "crates/airproxy-openai" }
airproxy-anthropic = { path = "crates/airproxy-anthropic" }
airproxy-stages    = { path = "crates/airproxy-stages" }
airproxy-config    = { path = "crates/airproxy-config" }
airproxy-http      = { path = "crates/airproxy-http" }

# NEW
ai-engine-core      = { path = "crates/ai-engine-core" }
ai-engine-provider  = { path = "crates/ai-engine-provider" }
ai-engine-openai    = { path = "crates/ai-engine-openai" }
ai-engine-anthropic = { path = "crates/ai-engine-anthropic" }
ai-engine-stages    = { path = "crates/ai-engine-stages" }
ai-engine-config    = { path = "crates/ai-engine-config" }
ai-engine-http      = { path = "crates/ai-engine-http" }
```

Use `sed -i` to do this safely:

```bash
sed -i 's/airproxy-core/ai-engine-core/g;
        s/airproxy-provider/ai-engine-provider/g;
        s/airproxy-openai/ai-engine-openai/g;
        s/airproxy-anthropic/ai-engine-anthropic/g;
        s/airproxy-stages/ai-engine-stages/g;
        s/airproxy-config/ai-engine-config/g;
        s/airproxy-http/ai-engine-http/g' Cargo.toml
```

`[workspace] members = ["crates/*"]` does not need editing — glob picks up the renamed dirs automatically.

- [ ] **Step 2: Verify no `airproxy` left in root Cargo.toml**

```bash
grep -n "airproxy" Cargo.toml
# Expected: no output (exit code 1 from grep)
```

If grep finds anything (e.g., a comment), inspect and decide per case. The repository field still says `https://github.com/alessiog/airproxy` — keep as-is for now since the GitHub repo doesn't exist yet; we'll update when the repo is created. Add a note in the commit message.

Actually, the `repository` field is misleading — change it too:

```bash
sed -i 's|github.com/alessiog/airproxy|github.com/alessiog/ai-engine|g' Cargo.toml
```

- [ ] **Step 3: No commit yet — cargo still won't build (per-crate Cargo.tomls and source imports still use old names).**

---

### Task 4: Update per-crate `Cargo.toml` files

**Files:**
- Modify: `crates/ai-engine/Cargo.toml`
- Modify: `crates/ai-engine-core/Cargo.toml`
- Modify: `crates/ai-engine-provider/Cargo.toml`
- Modify: `crates/ai-engine-openai/Cargo.toml`
- Modify: `crates/ai-engine-anthropic/Cargo.toml`
- Modify: `crates/ai-engine-stages/Cargo.toml`
- Modify: `crates/ai-engine-config/Cargo.toml`
- Modify: `crates/ai-engine-http/Cargo.toml`

Each crate's `Cargo.toml` has:
1. `[package].name = "airproxy-..."` — must be renamed.
2. References to other internal crates as deps (`airproxy-provider.workspace = true`) — must be renamed.
3. The `[lib]` or `[[bin]]` sections in `crates/ai-engine/Cargo.toml` reference `name = "airproxy"` — must be renamed.

- [ ] **Step 1: Do a single sed pass over every per-crate Cargo.toml**

```bash
cd /home/alessio/aip/airproxy
find crates -name Cargo.toml -print0 | xargs -0 sed -i '
    s/^name = "airproxy"$/name = "ai-engine"/g;
    s/^name = "airproxy-core"$/name = "ai-engine-core"/g;
    s/^name = "airproxy-provider"$/name = "ai-engine-provider"/g;
    s/^name = "airproxy-openai"$/name = "ai-engine-openai"/g;
    s/^name = "airproxy-anthropic"$/name = "ai-engine-anthropic"/g;
    s/^name = "airproxy-stages"$/name = "ai-engine-stages"/g;
    s/^name = "airproxy-config"$/name = "ai-engine-config"/g;
    s/^name = "airproxy-http"$/name = "ai-engine-http"/g;
    s/^name = "airproxy"\(.*\)$/name = "ai-engine"\1/g;
    s/airproxy-core\.workspace/ai-engine-core.workspace/g;
    s/airproxy-provider\.workspace/ai-engine-provider.workspace/g;
    s/airproxy-openai\.workspace/ai-engine-openai.workspace/g;
    s/airproxy-anthropic\.workspace/ai-engine-anthropic.workspace/g;
    s/airproxy-stages\.workspace/ai-engine-stages.workspace/g;
    s/airproxy-config\.workspace/ai-engine-config.workspace/g;
    s/airproxy-http\.workspace/ai-engine-http.workspace/g;
    s|path = "src/main\.rs"|path = "src/main.rs"|g'
```

- [ ] **Step 2: Update the binary name in `crates/ai-engine/Cargo.toml`**

The `[[bin]]` section currently says `name = "airproxy"`. After the sed above it should already be renamed, but verify:

```bash
grep -A 2 "\[\[bin\]\]" crates/ai-engine/Cargo.toml
# Expected:
# [[bin]]
# name = "ai-engine"
# path = "src/main.rs"
```

If `name = "airproxy"` is still present, fix manually with a precise Edit.

- [ ] **Step 3: Confirm no `airproxy` left in any Cargo.toml under crates/**

```bash
grep -rn "airproxy" crates/*/Cargo.toml
# Expected: no output.
```

- [ ] **Step 4: cargo check should still fail — source files still import `airproxy_*`**

```bash
cargo check --workspace 2>&1 | tail -10
# Expected: dozens of errors from `use airproxy_core::*` etc.
# This is the next task.
```

- [ ] **Step 5: No commit yet.**

---

### Task 5: Update source-file imports and identifiers

**Files:** every `.rs` file in `crates/*/src/` and `crates/*/tests/`.

The substitution rule: `airproxy_<crate>` (with underscore — Rust module form) becomes `ai_engine_<crate>`. Note that Rust converts hyphens to underscores in identifiers; the crate is `ai-engine-core` but the import is `use ai_engine_core::...`. So sed targets the underscore form.

Also: hyphen-delimited project name in comments / doc strings / log lines.

- [ ] **Step 1: Mass-substitute `airproxy_*` (underscore form) identifiers**

```bash
cd /home/alessio/aip/airproxy
find crates -name '*.rs' -print0 | xargs -0 sed -i '
    s/airproxy_core/ai_engine_core/g;
    s/airproxy_provider/ai_engine_provider/g;
    s/airproxy_openai/ai_engine_openai/g;
    s/airproxy_anthropic/ai_engine_anthropic/g;
    s/airproxy_stages/ai_engine_stages/g;
    s/airproxy_config/ai_engine_config/g;
    s/airproxy_http/ai_engine_http/g;
    s/extern crate airproxy/extern crate ai_engine/g'
```

- [ ] **Step 2: Substitute hyphen-form `airproxy-*` in doc-comments and log strings**

These appear in `//! airproxy-X` doc-comments at the top of every `lib.rs`, and possibly in `tracing` log targets:

```bash
find crates -name '*.rs' -print0 | xargs -0 sed -i '
    s|//! airproxy-core|//! ai-engine-core|g;
    s|//! airproxy-provider|//! ai-engine-provider|g;
    s|//! airproxy-openai|//! ai-engine-openai|g;
    s|//! airproxy-anthropic|//! ai-engine-anthropic|g;
    s|//! airproxy-stages|//! ai-engine-stages|g;
    s|//! airproxy-config|//! ai-engine-config|g;
    s|//! airproxy-http|//! ai-engine-http|g;
    s|//! airproxy|//! ai-engine|g'
```

- [ ] **Step 3: Substitute plain `airproxy` references in source (CLI strings, defaults, comments)**

This is the riskiest substitution — `airproxy` as a bare word could appear in unrelated places. Use a targeted approach: only substitute on lines we expect to contain it (string literals, comments).

```bash
# Show every remaining occurrence first so we can audit
grep -rn "airproxy" crates/*/src/ crates/*/tests/
```

Inspect the output. Typical hits expected:
1. `clap` `#[command(name = "airproxy", ...)]` in `crates/ai-engine/src/cli.rs` — rename to `ai-engine`.
2. Default config path `"airproxy.toml"` in `crates/ai-engine/src/cli.rs` — rename to `ai-engine.toml`.
3. `println!("airproxy: gateway core (stub)")` in `crates/ai-engine/src/main.rs` if leftover — should already be gone since we replaced main.rs in Task 13 of the v0.1 plan, but check.
4. `tracing::info!(..., "airproxy listening")` in `crates/ai-engine/src/main.rs` — rename to `"ai-engine listening"`.
5. Various test fixture strings — leave unchanged if they don't affect behavior.

For each remaining hit, decide:
- **Behavior-affecting** (CLI name, config path, log message that operators grep for) → rename to `ai-engine`.
- **Comment / doc** → rename to `ai-engine` for consistency.
- **Test fixture string** → leave unchanged unless renaming improves clarity.

Apply renames with targeted `Edit` tool calls (the subagent should NOT do a blanket sed for these — they're context-dependent).

- [ ] **Step 4: cargo check should now succeed**

```bash
cargo check --workspace 2>&1 | tail -5
# Expected: "Finished `dev` profile [unoptimized + debuginfo] target(s) in X.Xs"
```

If `cargo check` fails, the most common cause is a leftover `airproxy_*` import in a `tests/` file that the find scope missed. Re-run:

```bash
grep -rn "airproxy" crates/
# Audit every result, fix the missed cases, re-check.
```

- [ ] **Step 5: No commit yet** — config file and docs still pending.

---

### Task 6: Rename config file + update references

**Files:**
- Move: `airproxy.toml.example` → `ai-engine.toml.example`
- Modify: `crates/ai-engine/src/cli.rs` (default config path)
- Modify: `README.md` (every reference to `airproxy.toml`)
- Modify: `airproxy.toml.example` content (it mentions the project name in its header comment)
- Modify: per-task test fixtures that reference the old config file name (if any)

- [ ] **Step 1: Rename the example file**

```bash
cd /home/alessio/aip/airproxy
git mv airproxy.toml.example ai-engine.toml.example
```

- [ ] **Step 2: Update default config path in CLI**

`crates/ai-engine/src/cli.rs` currently has:

```rust
#[arg(short, long, default_value = "airproxy.toml")]
pub config: PathBuf,
```

Change to:

```rust
#[arg(short, long, default_value = "ai-engine.toml")]
pub config: PathBuf,
```

Use the Edit tool — don't sed, since the surrounding context matters.

- [ ] **Step 3: Sweep README and example file for `airproxy.toml` references**

```bash
grep -rn "airproxy\.toml" /home/alessio/aip/airproxy/
```

For each hit, replace with `ai-engine.toml`. Locations expected:
- `README.md` (quickstart, ollama section)
- `ai-engine.toml.example` (header comment, but it doesn't reference its own filename — check anyway)

```bash
sed -i 's/airproxy\.toml/ai-engine.toml/g' README.md
sed -i 's/airproxy\.toml/ai-engine.toml/g' ai-engine.toml.example
```

- [ ] **Step 4: Sweep README for bare "airproxy" word**

```bash
grep -n "airproxy" README.md
```

Replace every operationally-meaningful hit with `ai-engine`. The README references:
- The binary name in CLI examples (`./target/release/airproxy` → `./target/release/ai-engine`)
- The project name in prose
- The workspace member list
- The URL path comments

Apply with targeted Edit calls — each occurrence is context-dependent.

- [ ] **Step 5: cargo build the binary and confirm renamed binary exists**

```bash
cargo build --workspace --release 2>&1 | tail -3
ls target/release/ai-engine
# Expected: file exists.
ls target/release/airproxy 2>&1
# Expected: "No such file or directory" — old name purged.
```

- [ ] **Step 6: Smoke-test the renamed binary**

```bash
OPENAI_API_KEY=x ANTHROPIC_API_KEY=x AIRPROXY_MASTER_KEY=x \
  ./target/release/ai-engine --check --config ai-engine.toml.example
# Expected: "config OK: ai-engine.toml.example"
```

Note: `AIRPROXY_MASTER_KEY` is the env var name used in the example file (since v0.1). Renaming it is *part of the rename surface area*. Update the example to use `AI_ENGINE_MASTER_KEY` and document this as a breaking change in the commit message. Apply:

```bash
sed -i 's/AIRPROXY_MASTER_KEY/AI_ENGINE_MASTER_KEY/g' ai-engine.toml.example
```

Re-run the smoke test with the new env var:

```bash
OPENAI_API_KEY=x ANTHROPIC_API_KEY=x AI_ENGINE_MASTER_KEY=x \
  ./target/release/ai-engine --check --config ai-engine.toml.example
# Expected: "config OK: ai-engine.toml.example"
```

- [ ] **Step 7: No commit yet — final verification + docs sweep next.**

---

### Task 7: Final verification — tests, clippy, residual-name audit

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

```bash
cd /home/alessio/aip/airproxy
cargo test --workspace 2>&1 | tee /tmp/after-rename-tests.txt
grep -E "^test result:" /tmp/after-rename-tests.txt
```

Compare against `/tmp/before-rename-tests.txt`. Pass counts must match. If any test fails, the most likely cause is:
- A stale `use airproxy_*` import in a test file that got missed → grep for `airproxy` in `tests/` and fix.
- A test that asserts on a CLI string we changed (e.g., `cli.rs` clap name) → update the test to assert on the new name. This is the only legitimate test change in this plan.

- [ ] **Step 2: Run clippy**

```bash
cargo clippy --workspace --all-targets -- -D warnings 2>&1 | tee /tmp/after-rename-clippy.txt
echo $?
# Expected: 0
```

- [ ] **Step 3: Audit for any remaining `airproxy` strings**

```bash
grep -rn "airproxy" /home/alessio/aip/airproxy/ --include='*.rs' --include='*.toml' --include='*.md' --include='*.example' \
    | grep -v "^/home/alessio/aip/airproxy/docs/superpowers/specs/2026-05-23-airproxy-gateway-core-design.md:" \
    | grep -v "^/home/alessio/aip/airproxy/docs/superpowers/plans/2026-05-23-airproxy-gateway-core.md:"
```

Expected output: **empty**. Historical spec / plan documents from v0.1 are excluded (they describe the project as it was then; we're not rewriting history).

If non-empty output appears, inspect each line. Common false positives:
- Historical commit-message text in plans/specs that reference v0.1 — leave unchanged.
- The `repository` URL in some Cargo.toml that got missed — fix.
- A docstring quoting the old name in narrative context — fix.

- [ ] **Step 4: Audit for any remaining `AIRPROXY_` env-var name**

```bash
grep -rn "AIRPROXY_" /home/alessio/aip/airproxy/ \
    | grep -v "^/home/alessio/aip/airproxy/docs/superpowers/"
```

Expected: empty. If anything appears in `tests/`, those are command lines that should be updated to use `AI_ENGINE_MASTER_KEY` (or whatever the relevant new var is).

- [ ] **Step 5: Confirm test counts match baseline**

```bash
PASSED_BEFORE=$(grep -oE "[0-9]+ passed" /tmp/before-rename-tests.txt | awk '{sum += $1} END {print sum}')
PASSED_AFTER=$(grep -oE "[0-9]+ passed" /tmp/after-rename-tests.txt | awk '{sum += $1} END {print sum}')
echo "BEFORE: $PASSED_BEFORE; AFTER: $PASSED_AFTER"
test "$PASSED_BEFORE" = "$PASSED_AFTER" && echo "OK" || echo "REGRESSION"
```

Expected: `OK`. If `REGRESSION`, do not commit — diagnose first.

---

### Task 8: Commit the rename and tag

**Files:** none (git operations only)

- [ ] **Step 1: Review the diff**

```bash
git status
git diff --stat
```

Confirm:
- 8 crate dirs renamed.
- `airproxy.toml.example` renamed to `ai-engine.toml.example`.
- ~50–80 source files modified (mostly imports + doc comments).
- Root `Cargo.toml`, every per-crate `Cargo.toml`, `README.md` modified.

No `*.bak` or stray files.

- [ ] **Step 2: Single atomic commit**

```bash
git add -A
git commit -m "chore: rename airproxy to ai-engine

Mechanical rename across the workspace ahead of v0.2's distributed-
inference work. v0.1 behavior is unchanged — all tests pass with the
same counts as before; clippy clean.

Breaking changes for operators:
- binary:    airproxy           -> ai-engine
- config:    airproxy.toml      -> ai-engine.toml
- env var:   AIRPROXY_MASTER_KEY -> AI_ENGINE_MASTER_KEY
- crates:    airproxy-*         -> ai-engine-*

Historical v0.1 spec and plan documents under docs/superpowers/ are
left unchanged as point-in-time records."
```

NO Co-Authored-By footer (per global preference).

- [ ] **Step 3: Tag v0.1.1**

```bash
git tag v0.1.1
git log --oneline -3
git tag
# Expected: v0.1.0 and v0.1.1 both present.
```

The rename is a breaking change for operators (binary name, config file, env var, crate names) but introduces no functional changes; bumping the patch version to v0.1.1 is appropriate. The next sub-project's first commit will be on v0.2.0-dev.

- [ ] **Step 4: Final smoke verification on the tagged commit**

```bash
OPENAI_API_KEY=x ANTHROPIC_API_KEY=x AI_ENGINE_MASTER_KEY=x \
  ./target/release/ai-engine --check --config ai-engine.toml.example
# Expected: "config OK: ai-engine.toml.example"
cargo test --workspace 2>&1 | grep -E "^test result:"
# Expected: same pass counts as baseline
```

---

## Self-review

**Spec coverage:** Plan 0 implements the "P0 — Prerequisite rename" item from §10 of the design spec. The spec lists every item this plan touches (crate names, binary, config file, env-var names, README, CLI help, file headers). Coverage is complete.

**Placeholder scan:** Run mentally — no "TBD" / "TODO" / "implement later" / "add error handling" in the plan. The one judgment call ("each remaining occurrence is context-dependent") is bounded by an explicit grep audit step that lists the occurrences for the implementer to decide on.

**Type consistency:** No types defined in this plan. Identifier renames are consistent throughout — `airproxy-core` always becomes `ai-engine-core` (and `airproxy_core` always becomes `ai_engine_core`); never abbreviated to `aiengine-core` or `ai_engine` etc.

**Risk acknowledged:**

- **The grep-driven audit in Tasks 5 and 7 is the safety net.** A pure sed pass would miss context-dependent renames (CLI help strings, test fixture data, log targets that need careful escaping). The plan deliberately mixes mass-sed for safe substitutions with `Edit`-tool surgery for context-sensitive ones, gated by explicit grep audits.
- **Test count parity is the correctness gate.** Baseline captured in Task 1, verified in Task 7 step 5. A mismatch fails the rename outright.
- **Historical documents are explicitly excluded** from the rename — they describe v0.1 as it shipped under the `airproxy` name and renaming them would rewrite history.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-23-plan-0-rename-airproxy-to-ai-engine.md`. Two execution options:

**1. Subagent-Driven (recommended)** — fresh subagent per task, review between tasks. Same workflow that shipped v0.1's 15 tasks cleanly.

**2. Inline Execution** — execute tasks in this session with checkpoints.

Plan 0 is ~30–60 minutes of mechanical work. Subagent-driven is overkill for a rename this size; inline execution with a single careful pass is probably the right call here. But the choice is yours.
