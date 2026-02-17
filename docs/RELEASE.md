# DeepSeek CLI Release Guide

## Goals
- Produce immutable binaries for Linux, macOS, and Windows.
- Publish checksums for every artifact.
- Keep downgrade path available through versioned release assets.

## Artifact contract
Each tagged release (`vX.Y.Z`) publishes:
- `deepseek-x86_64-unknown-linux-gnu.tar.gz`
- `deepseek-x86_64-apple-darwin.tar.gz`
- `deepseek-x86_64-pc-windows-msvc.zip`
- `checksums.txt`

All assets are generated from `cargo build --release --bin deepseek`.

## Release steps
1. Ensure CI is green on default branch.
2. Bump version metadata as needed.
3. Create and push tag:
   - `git tag vX.Y.Z`
   - `git push origin vX.Y.Z`
4. GitHub Actions `release.yml` builds artifacts and creates the release.
5. Validate checksums and run installer smoke checks.

## Installer smoke checks
macOS/Linux:

```bash
bash scripts/install.sh --dry-run --version vX.Y.Z
```

Windows:

```powershell
./scripts/install.ps1 -DryRun -Version vX.Y.Z
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
