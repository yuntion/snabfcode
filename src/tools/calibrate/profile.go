package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"runtime"
	"runtime/debug"
	"slices"
	"sort"
	"strconv"
	"strings"
)

type shape struct {
	N int      `json:"n"`
	Q []uint64 `json:"q"`
	P []uint64 `json:"p"`
}
type benchCase struct {
	shape
	Ell      int     `json:"ell"`
	Op       string  `json:"op"`
	Shifts   []int   `json:"shifts"`
	Patterns [][]int `json:"patterns"`
}
type observation struct {
	K       int `json:"truncated_limbs_k"`
	Skipped int `json:"full_blocks_skipped"`
	Net     int `json:"measured_single_limb_ntts_saved"`
}
type point struct {
	Shape        int           `json:"shape"`
	Ell          int           `json:"ell"`
	Op           string        `json:"op"`
	Shifts       []int         `json:"shifts"`
	K            []int         `json:"k"`
	Rows         []int         `json:"block_rows"`
	Samples      []string      `json:"samples_us"`
	Holdout      []string      `json:"holdout_us"`
	Median       string        `json:"median_us"`
	IQR          string        `json:"iqr_us"`
	Observations []observation `json:"observed_non_native"`
}
type profile struct {
	Schema     string            `json:"schema"`
	Units      string            `json:"units"`
	ABI        string            `json:"abi"`
	HardwareID string            `json:"hardware_id"`
	Hardware   map[string]string `json:"hardware"`
	Toolchain  map[string]string `json:"toolchain"`
	BenchRev   string            `json:"bench_rev"`
	Created    string            `json:"created_utc"`
	Samples    int               `json:"samples"`
	Warmup     int               `json:"warmup"`
	Holdout    int               `json:"holdout"`
	Quantiles  string            `json:"quantiles"`
	Shapes     []shape           `json:"q_p_shape"`
	Sites      []point           `json:"sites"`
	Digest     string            `json:"profile_sha256,omitempty"`
}

// Typed canonical payload, independent of JSON whitespace, escaping, field
// order and number-printer differences. No nulls, floats or bools in v1.
func feed(h io.Writer, x any) error {
	switch t := x.(type) {
	case map[string]any:
		keys := make([]string, 0, len(t))
		for k := range t {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		io.WriteString(h, "{")
		for _, k := range keys {
			feed(h, k)
			if err := feed(h, t[k]); err != nil {
				return err
			}
		}
		io.WriteString(h, "}")
	case []any:
		io.WriteString(h, "[")
		for _, v := range t {
			if err := feed(h, v); err != nil {
				return err
			}
		}
		io.WriteString(h, "]")
	case string:
		fmt.Fprintf(h, "s%d:%s", len(t), t)
	case json.Number:
		v, err := t.Int64()
		if err != nil {
			return err
		}
		fmt.Fprintf(h, "i%d;", v)
	default:
		return fmt.Errorf("non-profile JSON value %T", x)
	}
	return nil
}
func digest(x any) (string, error) {
	b, err := json.Marshal(x)
	if err != nil {
		return "", err
	}
	d := json.NewDecoder(bytes.NewReader(b))
	d.UseNumber()
	var tree any
	if err = d.Decode(&tree); err != nil {
		return "", err
	}
	h := sha256.New()
	if err = feed(h, tree); err != nil {
		return "", err
	}
	return fmt.Sprintf("%x", h.Sum(nil)), nil
}

// Durations are serialized as decimal microseconds. Quarter-nanosecond ticks
// represent type-7 sample quartiles (including an even-N median) exactly.
func us(ticks int64) string { return fmt.Sprintf("%d.%05d", ticks/4000, (ticks%4000)*25) }
func summarize(ns []int64) (raw []string, median, iqr string) {
	for _, t := range ns {
		raw = append(raw, us(4*t))
	}
	a := slices.Sorted(slices.Values(ns))
	q := func(j int) int64 {
		x := (len(a) - 1) * j
		i, r := x/4, int64(x%4)
		return (4-r)*a[i] + r*a[min(i+1, len(a)-1)]
	}
	return raw, us(q(2)), us(q(3) - q(1))
}

func field(data, key string) string {
	for line := range strings.SplitSeq(data, "\n") {
		a, b, ok := strings.Cut(line, ":")
		if ok && strings.TrimSpace(a) == key {
			return strings.TrimSpace(b)
		}
	}
	return "unavailable"
}
func readOr(path string) string {
	b, e := os.ReadFile(path)
	if e != nil {
		return "unavailable"
	}
	return strings.TrimSpace(string(b))
}
func hardware() (map[string]string, string, error) {
	status := readOr("/proc/thread-self/status")
	cpu := field(status, "Cpus_allowed_list")
	c, err := strconv.Atoi(cpu)
	if runtime.GOOS != "linux" || err != nil || c < 0 {
		return nil, "", fmt.Errorf("single Linux CPU affinity required; use taskset (observed %q)", cpu)
	}
	var stanza string
	for _, v := range strings.Split(readOr("/proc/cpuinfo"), "\n\n") {
		if field(v, "processor") == cpu {
			stanza = v
			break
		}
	}
	model := field(stanza, "model name")
	if model == "unavailable" {
		model = field(stanza, "CPU part")
	}
	if model == "unavailable" {
		return nil, "", fmt.Errorf("missing CPU model for reserved CPU %d", c)
	}
	id := readOr("/etc/machine-id")
	scope := "machine-id"
	if id == "unavailable" || id == "" || id == "uninitialized" {
		// Some WSL/container images have no machine-id. A real boot ID is
		// deliberately more restrictive: the receipt expires on reboot.
		id, scope = readOr("/proc/sys/kernel/random/boot_id"), "boot-id"
	}
	if id == "unavailable" || id == "" {
		return nil, "", fmt.Errorf("cannot identify benchmark host")
	}
	hw := map[string]string{"arch": runtime.GOARCH, "os": runtime.GOOS, "id_scope": scope, "machine_sha256": fmt.Sprintf("%x", sha256.Sum256([]byte(id))), "cpu_model": model, "microcode": field(stanza, "microcode"), "affinity": cpu, "governor": readOr(fmt.Sprintf("/sys/devices/system/cpu/cpu%d/cpufreq/scaling_governor", c))}
	h, e := digest(hw)
	return hw, h, e
}
func toolchain(override string) (map[string]string, string, error) {
	b, ok := debug.ReadBuildInfo()
	if !ok {
		return nil, "", fmt.Errorf("missing Go build info")
	}
	tc := map[string]string{"go": runtime.Version(), "lattigo": "", "gomaxprocs": "1", "gc": "forced-before-each-call; disabled-inside-clock", "build": b.String()}
	rev := ""
	for _, d := range b.Deps {
		if d.Path == "github.com/tuneinsight/lattigo/v6" && d.Replace == nil {
			tc["lattigo"] = d.Version
		}
	}
	for _, s := range b.Settings {
		if s.Key == "vcs.revision" {
			rev = s.Value
		}
	}
	if override != "" {
		if rev != "" && rev != override {
			return nil, "", fmt.Errorf("--bench-rev disagrees with build VCS revision")
		}
		rev = override
	}
	if len(rev) != 40 || strings.Trim(rev, "0123456789abcdef") != "" || tc["lattigo"] != "v6.2.0" {
		return nil, "", fmt.Errorf("requires locked Lattigo v6.2.0 and a 40-hex build revision (use --bench-rev for source archives)")
	}
	path, e := os.Executable()
	if e != nil {
		return nil, "", e
	}
	f, e := os.Open(path)
	if e != nil {
		return nil, "", e
	}
	defer f.Close()
	h := sha256.New()
	if _, e = io.Copy(h, f); e != nil {
		return nil, "", e
	}
	tc["binary_sha256"] = fmt.Sprintf("%x", h.Sum(nil))
	return tc, rev, nil
}
