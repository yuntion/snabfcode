package epochs

import (
	"fmt"
	"runtime"
)

// Upstream HEIR panics on Lattigo error returns. Keep the generated public
// Eval error ABI without hiding programmer errors (bounds, nil dereferences).
func HEIRError(site *string, err *error) {
	if v := recover(); v != nil {
		if e, ok := v.(error); ok {
			if _, fault := e.(runtime.Error); !fault {
				*err = fmt.Errorf("%s: %w", *site, e)
				return
			}
		}
		panic(v)
	}
}
