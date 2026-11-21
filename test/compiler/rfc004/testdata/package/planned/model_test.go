package planned

import "testing"

func TestEval(t *testing.T) {
	if got := Eval(40); got != 42 {
		t.Fatalf("Eval(40) = %d", got)
	}
}
