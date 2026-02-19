# DeepSeek CLI Release Guide

## Goals
- Produce immutable binaries for Linux, macOS, and Windows.
- Publish checksums for every artifact.
- Generate SBOM per release.
- Generate provenance attestation for release artifacts.
- Publish package-manager update paths (Homebrew + Winget).
- Keep downgrade path available through versioned release assets.

## Artifact contract
Each tagged release (`vX.Y.Z`) publishes:
- `deepseek-x86_64-unknown-linux-gnu.tar.gz`
- `deepseek-aarch64-unknown-linux-gnu.tar.gz`
- `deepseek-x86_64-apple-darwin.tar.gz`
- `deepseek-aarch64-apple-darwin.tar.gz`
- `deepseek-x86_64-pc-windows-msvc.zip`
- `deepseek-aarch64-pc-windows-msvc.zip`
- `checksums.txt`
- `sbom.spdx.json`
- provenance attestation (GitHub artifact attestation)

All assets are generated from `cargo build --release --bin deepseek`.

## Release steps
1. Ensure CI is green on default branch (`ci.yml`, `replay-regression.yml`, `performance-gates.yml`, `security-gates.yml`).
2. Bump version metadata as needed.
3. Run manual release drill (`release-readiness.yml`) and confirm installer dry-runs + rollback rehearsal artifacts.
4. Create and push tag:
   - `git tag vX.Y.Z`
   - `git push origin vX.Y.Z`
5. GitHub Actions `release.yml` builds artifacts, SBOM, and provenance attestation, then creates the release.
6. Validate checksums and run installer smoke checks.
7. Homebrew tap update workflow (`homebrew.yml`) publishes formula update.
8. Winget update workflow (`winget.yml`) publishes manifest update.

## Installer smoke checks
macOS/Linux:

```bash
bash scripts/install.sh --dry-run --version vX.Y.Z
```

Windows:

```powershell
./scripts/install.ps1 -DryRun -Version vX.Y.Z
```

## Live API readiness checks
- Run `live-deepseek-smoke.yml` with `DEEPSEEK_API_KEY` secret configured.
- Verify:
  - `ask`, `plan`, and bounded `autopilot` commands succeed end-to-end.
  - first-token streaming latency gate remains within budget.

Package manager smoke checks:

```bash
brew install deepseek
brew uninstall deepseek
```

```powershell
winget install DeepSeek.DeepSeekCLI
winget uninstall DeepSeek.DeepSeekCLI
```

## Manual local packaging (fallback)
Linux/macOS:

```bash
cargo build --release --bin deepseek
cp target/release/deepseek dist/deepseek
```

Windows:

```powershell
cargo build --release --bin deepseek
Copy-Item target/release/deepseek.exe dist/deepseek.exe
```
