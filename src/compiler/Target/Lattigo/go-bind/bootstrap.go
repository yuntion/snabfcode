package main

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"os"
	"strconv"
	"strings"
)

// Preserve HEIR's encryption/configuration function and Bootstrap call. Bind
// only its evaluator package and the verified plan at its unique constructor.
func emitBootstrap(a []string) error {
	var audit struct {
		Bootstrap struct {
			Enabled             bool
			Execution, Function string
			N                   int `json:"source_n"`
			Target              int `json:"target_n"`
			Period              int `json:"source_period"`
			Level               int `json:"input_level"`
			Scale               int `json:"log_scale"`
			Q, P                []uint64
		}
	}
	if err := read(a[1], &audit); err != nil {
		return err
	}
	p := audit.Bootstrap
	if !p.Enabled || p.Execution != "periodic-n15-v1" || p.Function == "" {
		return fmt.Errorf("no checked executable ring bootstrap")
	}
	fs := token.NewFileSet()
	f, err := parser.ParseFile(fs, a[2], nil, 0)
	if err != nil {
		return err
	}
	imports, configs, calls := 0, 0, 0
	for _, im := range f.Imports {
		path, _ := strconv.Unquote(im.Path.Value)
		if path == "github.com/tuneinsight/lattigo/v6/circuits/ckks/bootstrapping" && im.Name == nil {
			im.Name = ast.NewIdent("bootstrapping")
			im.Path.Value = strconv.Quote("snabf.local/compiler/src/backend/ringswitch")
			imports++
		}
	}
	literal := fmt.Sprintf("bootstrapping.Plan{Source:%d,Target:%d,Period:%d,Level:%d,LogScale:%d,Q:%#v,P:%#v}", p.N, p.Target, p.Period, p.Level, p.Scale, p.Q, p.P)
	plan, err := parser.ParseExpr(literal)
	if err != nil {
		return err
	}
	for _, decl := range f.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok {
			continue
		}
		ast.Inspect(fn.Body, func(n ast.Node) bool {
			if e, ok := n.(ast.Expr); ok {
				if c, ok := method(e, "bootstrapping", "NewParametersFromLiteral", 2); ok && strings.EqualFold(fn.Name.Name, p.Function+"__configure") {
					c.Args = append(c.Args, plan)
					configs++
				}
			}
			if c, ok := n.(*ast.CallExpr); ok && strings.EqualFold(fn.Name.Name, p.Function) {
				if s, ok := c.Fun.(*ast.SelectorExpr); ok && s.Sel.Name == "Bootstrap" && len(c.Args) == 1 {
					calls++
				}
			}
			return true
		})
	}
	if imports != 1 || configs != 1 || calls != 1 {
		return fmt.Errorf("HEIR ring bootstrap binding mismatch: %d/%d/%d", imports, configs, calls)
	}
	var buf bytes.Buffer
	if err = format.Node(&buf, fs, f); err != nil {
		return err
	}
	return os.WriteFile(a[3], buf.Bytes(), 0644)
}
