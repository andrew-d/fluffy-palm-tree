// Package safetensors is a pure-Go reader for the HuggingFace SafeTensors
// format. Only the stdlib is used.
//
// The format is:
//
//	uint64le N       // length of JSON header
//	[N]byte  header  // UTF-8 JSON object keyed by tensor name (plus an
//	                 //   optional "__metadata__" key with freeform metadata)
//	[...]byte data   // concatenated little-endian tensor bytes; each tensor
//	                 //   entry's "data_offsets" is a [start, end] pair
//	                 //   relative to the start of this block.
package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
	"syscall"
)

// DType identifies the element type of a tensor. Only the dtypes we actually
// load from the privacy-filter weights are defined; add more as needed.
type DType int

const (
	// BF16 is bfloat16 (2 bytes/elem, little-endian).
	BF16 DType = iota
	// F32 is IEEE-754 binary32 (4 bytes/elem, little-endian).
	F32
)

// String returns the SafeTensors spelling of the dtype.
func (d DType) String() string {
	switch d {
	case BF16:
		return "BF16"
	case F32:
		return "F32"
	default:
		return fmt.Sprintf("DType(%d)", int(d))
	}
}

// itemSize returns the number of bytes per element for the dtype.
func (d DType) itemSize() int {
	switch d {
	case BF16:
		return 2
	case F32:
		return 4
	default:
		return 0
	}
}

// parseDType maps a SafeTensors dtype string to its DType. Only the dtypes
// used by the privacy-filter weights are supported today.
func parseDType(s string) (DType, error) {
	switch s {
	case "BF16":
		return BF16, nil
	case "F32":
		return F32, nil
	default:
		return 0, fmt.Errorf("safetensors: unsupported dtype %q", s)
	}
}

// Tensor is a single named tensor in a SafeTensors file.
type Tensor struct {
	Name  string
	DType DType
	Shape []int
	// Bytes is the raw little-endian payload. When Reader was created from
	// an mmap, this is a view into the mmap region and is valid only until
	// Close is called.
	Bytes []byte
}

// Float32s decodes the tensor's raw bytes into a freshly allocated []float32.
// BF16 tensors are widened to F32; F32 tensors are copied verbatim.
func (t *Tensor) Float32s() []float32 {
	switch t.DType {
	case BF16:
		n := len(t.Bytes) / 2
		out := make([]float32, n)
		for i := 0; i < n; i++ {
			bf := uint32(t.Bytes[2*i]) | uint32(t.Bytes[2*i+1])<<8
			out[i] = math.Float32frombits(bf << 16)
		}
		return out
	case F32:
		n := len(t.Bytes) / 4
		out := make([]float32, n)
		for i := 0; i < n; i++ {
			bits := binary.LittleEndian.Uint32(t.Bytes[4*i:])
			out[i] = math.Float32frombits(bits)
		}
		return out
	default:
		return nil
	}
}

// headerEntry is the shape of each value in the JSON header (excluding the
// special "__metadata__" key).
type headerEntry struct {
	DType       string   `json:"dtype"`
	Shape       []int    `json:"shape"`
	DataOffsets [2]int64 `json:"data_offsets"`
}

// Reader provides random access to tensors inside a SafeTensors file.
type Reader struct {
	file      *os.File
	mmap      []byte // full file mapping; nil if we fell back to ReadAt
	data      []byte // view of the tensor-data region (post-header); only set when mmap'd
	dataStart int64  // byte offset where the tensor-data block starts
	dataSize  int64  // size of the tensor-data block
	tensors   map[string]headerEntry
	names     []string // insertion order (sorted by data_offsets.start for stability)
}

// Open opens path and parses its SafeTensors header. The file is memory-mapped
// when possible so that large weight files do not need to be fully loaded into
// RAM; Close releases the mapping.
func Open(path string) (*Reader, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}

	st, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, err
	}
	size := st.Size()
	if size < 8 {
		f.Close()
		return nil, fmt.Errorf("safetensors: file %s too small (%d bytes)", path, size)
	}

	// Read the 8-byte header length.
	var hdrBuf [8]byte
	if _, err := io.ReadFull(io.NewSectionReader(f, 0, 8), hdrBuf[:]); err != nil {
		f.Close()
		return nil, fmt.Errorf("safetensors: read header length: %w", err)
	}
	hdrLen := int64(binary.LittleEndian.Uint64(hdrBuf[:]))
	if hdrLen <= 0 || 8+hdrLen > size {
		f.Close()
		return nil, fmt.Errorf("safetensors: bogus header length %d (file size %d)", hdrLen, size)
	}

	// Read the JSON header.
	hdrJSON := make([]byte, hdrLen)
	if _, err := io.ReadFull(io.NewSectionReader(f, 8, hdrLen), hdrJSON); err != nil {
		f.Close()
		return nil, fmt.Errorf("safetensors: read header: %w", err)
	}

	// We can't decode directly into a map[string]headerEntry because the
	// "__metadata__" key has a different shape. Decode into
	// json.RawMessage first and dispatch per key.
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(hdrJSON, &raw); err != nil {
		f.Close()
		return nil, fmt.Errorf("safetensors: parse header: %w", err)
	}

	tensors := make(map[string]headerEntry, len(raw))
	for name, value := range raw {
		if name == "__metadata__" {
			continue
		}
		var e headerEntry
		if err := json.Unmarshal(value, &e); err != nil {
			f.Close()
			return nil, fmt.Errorf("safetensors: parse header entry %q: %w", name, err)
		}
		if e.DataOffsets[0] < 0 || e.DataOffsets[1] < e.DataOffsets[0] {
			f.Close()
			return nil, fmt.Errorf("safetensors: tensor %q has invalid data_offsets %v", name, e.DataOffsets)
		}
		tensors[name] = e
	}

	dataStart := 8 + hdrLen
	dataSize := size - dataStart

	// Validate offsets are within the data block and compute a stable
	// ordering (by data_offsets.start) for Names().
	names := make([]string, 0, len(tensors))
	for name, e := range tensors {
		if e.DataOffsets[1] > dataSize {
			f.Close()
			return nil, fmt.Errorf("safetensors: tensor %q offsets %v exceed data block (%d bytes)", name, e.DataOffsets, dataSize)
		}
		// Shape sanity: product * itemSize must match the declared range.
		dt, err := parseDType(e.DType)
		if err == nil {
			want := int64(dt.itemSize())
			for _, d := range e.Shape {
				if d < 0 {
					want = -1
					break
				}
				want *= int64(d)
			}
			if want >= 0 && want != e.DataOffsets[1]-e.DataOffsets[0] {
				f.Close()
				return nil, fmt.Errorf("safetensors: tensor %q: shape %v * %d bytes = %d, but data_offsets span = %d", name, e.Shape, dt.itemSize(), want, e.DataOffsets[1]-e.DataOffsets[0])
			}
		}
		names = append(names, name)
	}
	sort.Slice(names, func(i, j int) bool {
		return tensors[names[i]].DataOffsets[0] < tensors[names[j]].DataOffsets[0]
	})

	r := &Reader{
		file:      f,
		dataStart: dataStart,
		dataSize:  dataSize,
		tensors:   tensors,
		names:     names,
	}

	// Try to mmap the whole file read-only. If the mapping fails (e.g. on
	// a platform that doesn't support it for this file type), fall back to
	// reading tensor bytes on demand via ReadAt.
	if size > 0 {
		mapped, err := syscall.Mmap(int(f.Fd()), 0, int(size), syscall.PROT_READ, syscall.MAP_SHARED)
		if err == nil {
			r.mmap = mapped
			r.data = mapped[dataStart:]
		}
	}

	return r, nil
}

// Close releases resources held by the Reader. Any Tensor.Bytes slices
// returned while the Reader was open become invalid after Close.
func (r *Reader) Close() error {
	var firstErr error
	if r.mmap != nil {
		if err := syscall.Munmap(r.mmap); err != nil && firstErr == nil {
			firstErr = err
		}
		r.mmap = nil
		r.data = nil
	}
	if r.file != nil {
		if err := r.file.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
		r.file = nil
	}
	return firstErr
}

// Names returns the names of all tensors in the file, sorted by their data
// offset so iteration order is deterministic across runs.
func (r *Reader) Names() []string {
	out := make([]string, len(r.names))
	copy(out, r.names)
	return out
}

// Tensor returns the named tensor, or an error if it does not exist.
func (r *Reader) Tensor(name string) (*Tensor, error) {
	entry, ok := r.tensors[name]
	if !ok {
		return nil, fmt.Errorf("safetensors: tensor %q not found", name)
	}
	dt, err := parseDType(entry.DType)
	if err != nil {
		return nil, err
	}

	shape := make([]int, len(entry.Shape))
	copy(shape, entry.Shape)

	nbytes := entry.DataOffsets[1] - entry.DataOffsets[0]
	var buf []byte
	if r.data != nil {
		// Zero-copy view into the mmap region.
		buf = r.data[entry.DataOffsets[0]:entry.DataOffsets[1]:entry.DataOffsets[1]]
	} else {
		buf = make([]byte, nbytes)
		if _, err := r.file.ReadAt(buf, r.dataStart+entry.DataOffsets[0]); err != nil {
			return nil, fmt.Errorf("safetensors: read tensor %q: %w", name, err)
		}
	}

	return &Tensor{
		Name:  name,
		DType: dt,
		Shape: shape,
		Bytes: buf,
	}, nil
}
