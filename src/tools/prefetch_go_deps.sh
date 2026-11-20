#!/usr/bin/env bash
# Fetch the module graph pinned by this project; make the actual Lattigo source visible.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
cd "$root"
go="$root/extern/go/bin/go"
[[ -x $go ]] || { echo 'Install the pinned Go toolchain with bootstrap_linux.sh first.' >&2; exit 1; }
export GOPATH="$root/extern/gopath" GOMODCACHE="$root/extern/gopath/pkg/mod"
export GOCACHE="$root/extern/go-cache" GOTOOLCHAIN=local GOWORK=off

"$go" mod download all
"$go" mod verify
module=github.com/tuneinsight/lattigo/v6
info=$("$go" list -mod=readonly -m -f '{{.Version}} {{if .Replace}}replaced{{else}}{{.Dir}}{{end}}' "$module")
read -r version source <<< "$info"
[[ $source == "$GOMODCACHE/$module@$version" && -f $source/go.mod ]] || {
  echo "Expected the verified module source for $module@$version, got: $source" >&2; exit 1;
}
view="$root/extern/lattigo"
if [[ ! -e $view && ! -L $view ]]; then
  ln -sT -- "$(realpath --relative-to="$root/extern" "$source")" "$view"
elif [[ ! $view -ef $source ]]; then
  echo "Existing $view does not point at $source; leaving it untouched." >&2; exit 1
fi
printf '%s -> %s (Go build source)\n' "$view" "$source"
