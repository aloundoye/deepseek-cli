# CodingBuddy Release Guide

## Goals
- Produce immutable binaries for Linux, macOS, and Windows.
- Publish checksums for every artifact.
- Keep downgrade path available through versioned release assets.
- Be explicit about what is automated in this repo versus what is still manual.

## Artifact contract
Each GitHub release created by `release.yml` currently publishes:
- `codingbuddy-x86_64-unknown-linux-gnu.tar.gz`
- `codingbuddy-aarch64-unknown-linux-gnu.tar.gz`
- `codingbuddy-x86_64-apple-darwin.tar.gz`
- `codingbuddy-aarch64-apple-darwin.tar.gz`
- `codingbuddy-x86_64-pc-windows-msvc.zip`
- `codingbuddy-aarch64-pc-windows-msvc.zip`
- `checksums.txt`

All assets are generated from `cargo build --release --bin codingbuddy`.

## Current automation scope
Automated in this repo today:
- `ci.yml`
- `release.yml`
- `benchmark-live.yml` (manual/nightly live benchmark lane, non-blocking)

Not automated in this repo today:
- SBOM generation
- provenance attestation
- Homebrew publishing
- Winget publishing
- dedicated release-readiness/security workflow split

## Release steps
1. Ensure `ci.yml` is green on `main`.
2. Bump version metadata as needed.
3. Push the version bump to `main`.
4. Let `release.yml`:
   - verify the version is not already released
   - build platform artifacts
   - generate `checksums.txt`
   - create tag `vX.Y.Z`
   - publish the GitHub release
5. If provider credentials are configured, review or trigger `benchmark-live.yml` for a current live benchmark artifact and optional DeepSeek-vs-reference comparison.
6. Validate release artifacts and checksums.
7. Validate installer smoke checks.
8. If you maintain package-manager distribution outside this repo, publish those updates manually.

## Installer smoke checks
macOS/Linux:

```bash
bash scripts/install.sh --dry-run --version vX.Y.Z --repo aloundoye/codingbuddy
```

Windows:

```powershell
./scripts/install.ps1 -DryRun -Version vX.Y.Z -Repo aloundoye/codingbuddy
```

## Manual checks not automated by this repo
- Live API smoke:
  - If `DEEPSEEK_API_KEY` is available, run a bounded manual smoke in a disposable workspace before announcing the release.
  - `benchmark-live.yml` is a supporting signal, not a release gate replacement.
- Package manager publishing:
  - No Homebrew or Winget workflow exists in this repo today.
  - Any package-manager distribution is a manual follow-up outside the current automation contract.
- Supply-chain extras:
  - SBOM and provenance are not produced by `release.yml` today.

## Manual local packaging (fallback)
Linux/macOS:

```bash
cargo build --release --bin codingbuddy
cp target/release/codingbuddy dist/codingbuddy
```

Windows:

```powershell
cargo build --release --bin codingbuddy
Copy-Item target/release/codingbuddy.exe dist/codingbuddy.exe
```
