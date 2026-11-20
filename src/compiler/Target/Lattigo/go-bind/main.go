// Bind checked SN-ABF tower, scale and gadget decisions to Go emitted by HEIR.
// This shared target adapter does not choose plans or translate MLIR.
package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"math"
	"os"
	"strconv"
	"strings"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"snabf.local/compiler/src/backend/gadget"
	"snabf.local/compiler/src/backend/tower"
	"sort"
)

func read(path string, into any) error {
	b, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	return json.Unmarshal(b, into)
}

type coefficientScale struct {
	Level, LogScale int
	Scale           *float64 // absent on the original HEIR preprocessing sites
}

type routeBinding struct {
	tower.Route
	PolyScales []coefficientScale
	Relins     []gadget.Site
}

func ident(e ast.Expr) string {
	if x, ok := e.(*ast.Ident); ok {
		return x.Name
	}
	return ""
}

func method(e ast.Expr, pkg, name string, arity int) (*ast.CallExpr, bool) {
	c, ok := e.(*ast.CallExpr)
	if !ok || len(c.Args) != arity {
		return nil, false
	}
	s, ok := c.Fun.(*ast.SelectorExpr)
	return c, ok && ident(s.X) == pkg && s.Sel.Name == name
}

func intLiteral(e ast.Expr, expected int) bool {
	x, ok := e.(*ast.BasicLit)
	return ok && x.Kind == token.INT && x.Value == strconv.Itoa(expected)
}

// HEIR's encode.scale is an integral logarithm. Bind the actual positive CKKS
// scale from Lattigo IR before encoder.Encode consumes the plaintext. The input
// data, table slots, evaluator operations and output buffers stay HEIR's.
func bindCoefficientScales(fn *ast.FuncDecl, scales []coefficientScale) error {
	allocs := map[string]*ast.CallExpr{}
	i := 0
	for _, stmt := range fn.Body.List {
		a, ok := stmt.(*ast.AssignStmt)
		if !ok || len(a.Lhs) != 1 || len(a.Rhs) != 1 {
			continue
		}
		if c, yes := method(a.Rhs[0], "ckks", "NewPlaintext", 2); yes {
			allocs[ident(a.Lhs[0])] = c
			continue
		}
		target, ok := a.Lhs[0].(*ast.SelectorExpr)
		if !ok || target.Sel.Name != "Scale" {
			continue
		}
		alloc := allocs[ident(target.X)]
		if alloc == nil || i >= len(scales) {
			return fmt.Errorf("unbound physical CKKS scale at preprocessing encode %d", i)
		}
		b := scales[i]
		set, ok := method(a.Rhs[0], ident(alloc.Args[0]), "NewScale", 1)
		if !ok || !intLiteral(alloc.Args[1], b.Level) {
			return fmt.Errorf("plaintext level or parameter mismatch at encode %d", i)
		}
		pow, ok := method(set.Args[0], "math", "Pow", 2)
		if !ok || !intLiteral(pow.Args[0], 2) || !intLiteral(pow.Args[1], b.LogScale) {
			return fmt.Errorf("plaintext log-scale mismatch at encode %d", i)
		}
		if b.Scale != nil {
			s := *b.Scale
			if !(s > 0) || math.IsInf(s, 0) || math.Round(math.Log2(s)) != float64(b.LogScale) {
				return fmt.Errorf("invalid physical CKKS scale at encode %d", i)
			}
			set.Args[0] = &ast.BasicLit{Kind: token.FLOAT, Value: strconv.FormatFloat(s, 'e', -1, 64)}
		}
		delete(allocs, ident(target.X))
		i++
	}
	if i != len(scales) || len(allocs) != 0 {
		return fmt.Errorf("physical scale bindings incomplete: %d encodes, %d records", i, len(scales))
	}
	return nil
}

func run(a []string) error {
	if len(a) == 4 && a[0] == "emit-bootstrap" {
		return emitBootstrap(a)
	}
	if len(a) == 2 && a[0] == "key-size" {
		var literal ckks.ParametersLiteral
		if err := read(a[1], &literal); err != nil {
			return err
		}
		p, err := ckks.NewParametersFromLiteral(literal)
		if err != nil {
			return err
		}
		fmt.Println(rlwe.NewGaloisKey(p).BinarySize())
		return nil
	}
	if len(a) != 4 || a[0] != "emit" {
		return fmt.Errorf("usage: key-size profile.json | emit routes.json heir.go model.go")
	}
	var routes []routeBinding
	if err := read(a[1], &routes); err != nil {
		return err
	}
	fs := token.NewFileSet()
	f, err := parser.ParseFile(fs, a[2], nil, parser.ParseComments)
	if err != nil {
		return err
	}
	imports, sites := 0, 0
	var relins []gadget.Site
	pass3 := false
	var scales []coefficientScale
	for _, route := range routes {
		if route.Relins != nil {
			if pass3 {
				return fmt.Errorf("duplicate Pass 3 relin table")
			}
			pass3 = true
			relins = route.Relins
		}
		if route.PolyScales != nil {
			if scales != nil {
				return fmt.Errorf("multiple polynomial preprocessing manifests")
			}
			scales = route.PolyScales
		}
	}
	if pass3 {
		used := 0
		for _, d := range f.Decls {
			fn, ok := d.(*ast.FuncDecl)
			if !ok || !strings.HasSuffix(fn.Name.Name, "__preprocessed") {
				continue
			}
			ast.Inspect(fn.Body, func(n ast.Node) bool {
				c, ok := n.(*ast.CallExpr)
				if !ok {
					return true
				}
				sel, ok := c.Fun.(*ast.SelectorExpr)
				if !ok || (sel.Sel.Name != "Relinearize" && sel.Sel.Name != "RelinearizeNew") {
					return true
				}
				method := sel.Sel.Name
				ev := sel.X
				sel.X = ast.NewIdent("gadget")
				c.Args = append([]ast.Expr{ev}, c.Args...)
				c.Args = append(c.Args, &ast.IndexExpr{X: ast.NewIdent("snabfRelins"), Index: &ast.BasicLit{Kind: token.INT, Value: strconv.Itoa(used)}})
				sel.Sel.Name = method
				used++
				return true
			})
		}
		if used != len(relins) {
			return fmt.Errorf("relin binding mismatch: %d calls / %d sites", used, len(relins))
		}
		for _, d := range f.Decls {
			if g, ok := d.(*ast.GenDecl); ok && g.Tok == token.IMPORT {
				g.Specs = append(g.Specs, &ast.ImportSpec{Path: &ast.BasicLit{Kind: token.STRING, Value: strconv.Quote("snabf.local/compiler/src/backend/gadget")}})
				break
			}
		}
	}
	for _, spec := range f.Imports {
		if spec.Path.Value == `"github.com/tuneinsight/lattigo/v6/circuits/ckks/lintrans"` && spec.Name == nil {
			spec.Name = ast.NewIdent("lintrans")
			spec.Path.Value = `"snabf.local/compiler/src/backend/tower"`
			imports++
		}
	}
	for _, decl := range f.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok || !strings.HasSuffix(fn.Name.Name, "__preprocessing") {
			continue
		}
		if scales != nil {
			if err := bindCoefficientScales(fn, scales); err != nil {
				return err
			}
			scales = nil
		}
		ast.Inspect(fn.Body, func(n ast.Node) bool {
			c, ok := n.(*ast.CallExpr)
			if !ok {
				return true
			}
			sel, ok := c.Fun.(*ast.SelectorExpr)
			if !ok {
				return true
			}
			pkg, ok := sel.X.(*ast.Ident)
			if ok && pkg.Name == "lintrans" && sel.Sel.Name == "NewTransformation" && len(c.Args) == 2 {
				c.Args = append(c.Args, &ast.IndexExpr{X: ast.NewIdent("snabfRoutes"), Index: &ast.BasicLit{Kind: token.INT, Value: strconv.Itoa(sites)}})
				sites++
			}
			return true
		})
	}
	if imports != 1 || sites != len(routes) || sites == 0 || scales != nil {
		return fmt.Errorf("HEIR import/prepare mismatch: %d imports, %d sites, %d plans", imports, sites, len(routes))
	}
	var out bytes.Buffer
	if err = format.Node(&out, fs, f); err != nil {
		return err
	}
	out.WriteString("\nvar snabfRoutes = []lintrans.Route{\n")
	for _, r := range routes {
		fmt.Fprintf(&out, "{LogN:%d,Level:%d,LogSlots:%d,Terms:[][3]int{", r.LogN, r.Level, r.LogSlots)
		for _, t := range r.Terms {
			fmt.Fprintf(&out, "{%d,%d,%d},", t[0], t[1], t[2])
		}
		out.WriteString("},")
		if pass3 {
			for _, entry := range []struct {
				name string
				m    map[int]gadget.Site
			}{{"Babies", r.Babies}, {"Giants", r.Giants}} {
				if entry.m == nil {
					return fmt.Errorf("missing %s table", entry.name)
				}
				fmt.Fprintf(&out, "%s:map[int]gadget.Site{", entry.name)
				keys := []int{}
				for v := range entry.m {
					keys = append(keys, v)
				}
				sort.Ints(keys)
				for _, k := range keys {
					s := entry.m[k]
					fmt.Fprintf(&out, "%d:{ID:%q,Level:%d,K:%d},", k, s.ID, s.Level, s.K)
				}
				out.WriteString("},")
			}
		}
		out.WriteString("},\n")
	}
	out.WriteString("}\n")
	if pass3 {
		out.WriteString("var snabfRelins=[]gadget.Site{")
		for _, s := range relins {
			fmt.Fprintf(&out, "{ID:%q,Level:%d,K:%d},", s.ID, s.Level, s.K)
		}
		out.WriteString("}\n")
	}
	source, err := format.Source(out.Bytes())
	if err != nil {
		return err
	}
	return os.WriteFile(a[3], source, 0644)
}

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
