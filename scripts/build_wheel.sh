#!/bin/bash
# =============================
# Build Daft Wheel for Distribution
# =============================
#
# Input Parameters:
#   --os: [ubuntu|macos|windows]
#   --arch: [x86_64|aarch64]
#   --lts: [true|false]
#   --build-type: [release|dev|nightly]
#
# Environment Variables:
#   PYTHON_VERSION: 3.11
#   DAFT_ANALYTICS_ENABLED: 0
#   UV_SYSTEM_PYTHON: 1
#
# Output:
#   Generated wheel file in dist/ directory

set -euo pipefail

DIR="$(cd "`dirname "$0"`"/..; pwd)"

# ========================
# Initialize Environment
# ========================

NAME="daft"
VERSION=""

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --os) OS="$2"; shift ;;
        --arch) ARCH="$2"; shift ;;
        --lts) LTS="$2"; shift ;;
        --build-type) BUILD_TYPE="$2"; shift ;;
        -n|--name) NAME="$2"; shift ;;
        -v|--version) VERSION="$2"; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
    shift
done

# Validate required parameters
if [[ -z ${OS:-} || -z ${ARCH:-} || -z ${LTS:-} || -z ${BUILD_TYPE:-} ]]; then
    echo "Missing required parameters!"
    echo "Usage: $0 --os [ubuntu|macos|windows] --arch [x86_64|aarch64] --lts [true|false] --build-type [release|dev|nightly] [--name NAME] [--version VERSION]"
    echo "Defaults: --name daft"
    exit 1
fi

if ! [[ "$NAME" =~ ^("daft"|"ve-daft")$ ]]; then
    echo "Error: --lts must be 'daft' or 've-daft'"
    exit 2
fi

# Configure environment variables
PYTHON_VERSION=${PYTHON_VERSION:-"3.11"}
DAFT_ANALYTICS_ENABLED=${DAFT_ANALYTICS_ENABLED:-"0"}
UV_SYSTEM_PYTHON=${UV_SYSTEM_PYTHON:-"1"}
RUST_DAFT_PKG_BUILD_TYPE="$BUILD_TYPE"
RELEASE_DASHBOARD=${RELEASE_DASHBOARD:-"false"}

echo "========================================"
echo "Starting Build Process"
echo "========================================"
echo "Operating System: $OS"
echo "Architecture: $ARCH"
echo "LTS Build: $LTS"
echo "Build Type: $BUILD_TYPE"
echo "Name: $NAME"
echo "Version: $VERSION"
echo "========================================"

# =========================================
# Prepare Build Environment
# =========================================

# Setup Python version
echo "Setting up Python $PYTHON_VERSION..."
uv venv --seed -p "$PYTHON_VERSION"
source .venv/bin/activate
uv pip install -r $DIR/requirements-dev.txt

# Install Bun and required tools
if [[ "$RELEASE_DASHBOARD" == "true" ]]; then
  echo "Installing bun..."
  # Using Homebrew for macOS, apt for Ubuntu, choco for Windows
  case "$OS" in
      macos) brew install bun ;;
      ubuntu) sudo apt-get update && sudo apt-get install -y bun ;;
      windows) choco install bun ;;
  esac
else
  echo "Skip installing bun..."
fi

# Install Python dependencies
echo "Installing Python dependencies..."
uv pip install twine yq setuptools_scm

# =========================================
# Source Code Preparation
# =========================================

# Patch package version with setuptools_scm
VERSION=${VERSION:-$(python -m setuptools_scm | sed 's/\.dev/-dev/g')}
echo "Patching package version..."
echo "Setting package version to: $VERSION"

tomlq -i -t ".package.version = \"$VERSION\"" Cargo.toml
tomlq -i -t ".workspace.package.version = \"$VERSION\"" Cargo.toml


# Patch name for LTS builds
if [[ "$LTS" == "true" ]]; then
    echo "Patching project name to '$NAME-lts' for LTS build"
    tomlq -i -t ".project.name = \"$NAME-lts\"" pyproject.toml
else
    echo "Patching project name to '$NAME' for non-LTS build"
    tomlq -i -t ".project.name = \"$NAME\"" pyproject.toml
fi

# =========================================
# Configure Architecture-Specific Flags
# =========================================
if [[ "$ARCH" == "x86_64" ]]; then
    echo "Configuring x86_64-specific flags..."

    if [[ "$LTS" == "true" ]]; then
        export RUSTFLAGS="-C target-feature=+sse3,+ssse3,+sse4.1,+sse4.2,+popcnt,+cmpxchg16b"
        export CFLAGS="-msse3 -mssse3 -msse4.1 -msse4.2 -mpopcnt -mcx16"
    else
        export RUSTFLAGS="-C target-feature=+sse3,+ssse3,+sse4.1,+sse4.2,+popcnt,+cmpxchg16b,+avx,+avx2,+fma,+bmi1,+bmi2,+lzcnt,+pclmulqdq,+movbe -Z tune-cpu=skylake"
        export CFLAGS="-msse3 -mssse3 -msse4.1 -msse4.2 -mpopcnt -mcx16 -mavx -mavx2 -mfma -mbmi -mbmi2 -mlzcnt -mpclmul -mmovbe -mtune=skylake"
    fi

    echo "RUSTFLAGS set to: $RUSTFLAGS"
    echo "CFLAGS set to: $CFLAGS"
fi

# =========================================
# Build Dashboard Frontend
# =========================================
if [[ "$RELEASE_DASHBOARD" == "true" ]]; then
  echo "Building dashboard frontend..."
  pushd "./src/daft-dashboard/frontend" > /dev/null
  bun install
  bun run build
  popd > /dev/null
else
  echo "Skip building dashboard frontend..."
fi

# =========================================
# Build Wheels - Platform Specific
# =========================================
BUILD_ARGS="--profile release-lto --out dist"
SDIST_ARG="--sdist"

case "$OS" in
    # macOS Builds
    macos)
        if [[ "$ARCH" == "x86_64" ]]; then
            echo "Building macOS x86_64 wheel..."
            maturin build --target x86_64-apple-darwin $BUILD_ARGS
        elif [[ "$ARCH" == "aarch64" ]]; then
            echo "Building macOS aarch64 wheel..."
            export RUSTFLAGS="-Ctarget-cpu=apple-m1"
            export CFLAGS="-mtune=apple-m1"
            maturin build --target aarch64-apple-darwin $BUILD_ARGS
        fi
        ;;

    # Windows Builds
    windows)
        if [[ "$ARCH" == "x86_64" ]]; then
            echo "Building Windows x86_64 wheel..."
            maturin build --target x86_64-uwp-windows-gnu $BUILD_ARGS
        fi
        # Add ARM64 support for Windows here if needed
        ;;

    # Linux Builds
    ubuntu)
        case "$ARCH" in
            x86_64)
                echo "Building Linux x86_64 wheel (manylinux_2_24)..."
                maturin build --target x86_64-unknown-linux-gnu  $BUILD_ARGS $SDIST_ARG
                ;;
            aarch64)
                echo "Building Linux aarch64 wheel (manylinux_2_24)..."
                export JEMALLOC_SYS_WITH_LG_PAGE=16
                maturin build --target aarch64-unknown-linux-gnu $BUILD_ARGS $SDIST_ARG
                ;;
        esac
        ;;
esac

# =========================================
# Artifact Handling
# =========================================
ARTIFACT_NAME="wheels-$OS-$ARCH-lts=$LTS"

echo "========================================"
echo "Build Completed Successfully"
echo "========================================"
echo "Wheel files are available in: dist/"
echo "Artifact name: $ARTIFACT_NAME"
