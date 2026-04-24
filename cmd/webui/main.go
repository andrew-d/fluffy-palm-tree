// Command webui is a minimal web UI demonstrating the privacyfilter model.
//
// A single background goroutine services an in-memory FIFO queue, so at most
// one classification runs at a time. Clients submit text via POST /jobs, get
// a job ID back immediately, and then poll GET /jobs/{id} for queue position
// and, once complete, the classification results.
package main

import (
	"context"
	"embed"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io/fs"
	"log"
	"net/http"
	"os/signal"
	"sync"
	"syscall"
	"time"
	"unicode/utf8"

	"github.com/andrew-d/openai-privacy"
)

//go:embed static
var staticFS embed.FS

type jobStatus string

const (
	statusQueued jobStatus = "queued"
	statusActive jobStatus = "active"
	statusDone   jobStatus = "done"
	statusError  jobStatus = "error"
)

type job struct {
	id       string
	text     string
	status   jobStatus
	entities []privacyfilter.Entity
	errMsg   string
	// enqueuedAt is used only for debugging / ordering diagnostics.
	enqueuedAt time.Time
	// Throughput, set when status == done.
	duration time.Duration
	chars    int
	tokens   int
}

// queue is a minimal FIFO with a condition variable so the worker can sleep
// when empty. Kept intentionally simple: the worker holds the single slot,
// every other action goes through the mutex.
type queue struct {
	mu      sync.Mutex
	cond    *sync.Cond
	jobs    map[string]*job
	pending []string // job IDs, head = next to run
	nextID  uint64
	closed  bool
}

func newQueue() *queue {
	q := &queue{jobs: make(map[string]*job)}
	q.cond = sync.NewCond(&q.mu)
	return q
}

func (q *queue) enqueue(text string) *job {
	q.mu.Lock()
	defer q.mu.Unlock()
	q.nextID++
	id := fmt.Sprintf("j%d", q.nextID)
	j := &job{
		id:         id,
		text:       text,
		status:     statusQueued,
		enqueuedAt: time.Now(),
	}
	q.jobs[id] = j
	q.pending = append(q.pending, id)
	q.cond.Signal()
	return j
}

// pop blocks until a job is available or the queue is closed. Returns nil on
// close.
func (q *queue) pop() *job {
	q.mu.Lock()
	defer q.mu.Unlock()
	for len(q.pending) == 0 && !q.closed {
		q.cond.Wait()
	}
	if q.closed {
		return nil
	}
	id := q.pending[0]
	q.pending = q.pending[1:]
	j := q.jobs[id]
	j.status = statusActive
	return j
}

func (q *queue) close() {
	q.mu.Lock()
	q.closed = true
	q.cond.Broadcast()
	q.mu.Unlock()
}

// snapshot returns a copy of the job plus its 0-based queue position (or -1 if
// not queued). Position 0 means "next up"; the active job reports -1.
func (q *queue) snapshot(id string) (*job, int, bool) {
	q.mu.Lock()
	defer q.mu.Unlock()
	j, ok := q.jobs[id]
	if !ok {
		return nil, 0, false
	}
	pos := -1
	for i, qid := range q.pending {
		if qid == id {
			pos = i
			break
		}
	}
	// Return a shallow copy so callers can read fields without the lock.
	cp := *j
	return &cp, pos, true
}

func (q *queue) complete(id string, ents []privacyfilter.Entity, chars, tokens int, dur time.Duration, err error) {
	q.mu.Lock()
	defer q.mu.Unlock()
	j, ok := q.jobs[id]
	if !ok {
		return
	}
	if err != nil {
		j.status = statusError
		j.errMsg = err.Error()
		return
	}
	j.status = statusDone
	j.entities = ents
	j.chars = chars
	j.tokens = tokens
	j.duration = dur
}

// queueLen returns the number of pending (not-yet-started) jobs.
func (q *queue) queueLen() int {
	q.mu.Lock()
	defer q.mu.Unlock()
	return len(q.pending)
}

// --- HTTP handlers ---

type submitRequest struct {
	Text string `json:"text"`
}

type submitResponse struct {
	ID string `json:"id"`
}

type entityJSON struct {
	EntityGroup string  `json:"entity_group"`
	Score       float32 `json:"score"`
	Word        string  `json:"word"`
	Start       int     `json:"start"`
	End         int     `json:"end"`
}

type statusResponse struct {
	ID       string       `json:"id"`
	Status   jobStatus    `json:"status"`
	Position int          `json:"position"` // 0 = next; -1 = active/done/error
	Queued   int          `json:"queue_length"`
	Text     string       `json:"text,omitempty"`
	Entities []entityJSON `json:"entities,omitempty"`
	Error    string       `json:"error,omitempty"`
	// Timing and throughput, present only when status == done.
	DurationMs    float64 `json:"duration_ms,omitempty"`
	Chars         int     `json:"chars,omitempty"`
	Tokens        int     `json:"tokens,omitempty"`
	CharsPerSec   float64 `json:"chars_per_sec,omitempty"`
	TokensPerSec  float64 `json:"tokens_per_sec,omitempty"`
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v)
}

func handleSubmit(q *queue) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		var req submitRequest
		if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1<<20)).Decode(&req); err != nil {
			http.Error(w, "invalid JSON body", http.StatusBadRequest)
			return
		}
		if req.Text == "" {
			http.Error(w, "text must not be empty", http.StatusBadRequest)
			return
		}
		j := q.enqueue(req.Text)
		writeJSON(w, http.StatusAccepted, submitResponse{ID: j.id})
	}
}

func handleStatus(q *queue) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		id := r.PathValue("id")
		j, pos, ok := q.snapshot(id)
		if !ok {
			http.Error(w, "job not found", http.StatusNotFound)
			return
		}
		resp := statusResponse{
			ID:       j.id,
			Status:   j.status,
			Position: pos,
			Queued:   q.queueLen(),
		}
		switch j.status {
		case statusDone:
			resp.Text = j.text
			resp.Entities = toJSONEntities(j.entities)
			resp.Chars = j.chars
			resp.Tokens = j.tokens
			resp.DurationMs = float64(j.duration.Microseconds()) / 1000.0
			if secs := j.duration.Seconds(); secs > 0 {
				resp.CharsPerSec = float64(j.chars) / secs
				resp.TokensPerSec = float64(j.tokens) / secs
			}
		case statusError:
			resp.Error = j.errMsg
		}
		writeJSON(w, http.StatusOK, resp)
	}
}

func toJSONEntities(ents []privacyfilter.Entity) []entityJSON {
	out := make([]entityJSON, len(ents))
	for i, e := range ents {
		out[i] = entityJSON{
			EntityGroup: e.EntityGroup,
			Score:       e.Score,
			Word:        e.Word,
			Start:       e.Start,
			End:         e.End,
		}
	}
	return out
}

// --- Worker ---

func runWorker(ctx context.Context, q *queue, model *privacyfilter.Model) {
	for {
		j := q.pop()
		if j == nil {
			return
		}
		if ctx.Err() != nil {
			q.complete(j.id, nil, 0, 0, 0, ctx.Err())
			return
		}
		chars := utf8.RuneCountInString(j.text)
		tokens := model.TokenCount(j.text)
		start := time.Now()
		ents, err := model.Classify(j.text)
		dur := time.Since(start)
		q.complete(j.id, ents, chars, tokens, dur, err)
	}
}

func main() {
	addr := flag.String("addr", ":8080", "HTTP listen address")
	modelDir := flag.String("model", "./model", "path to privacy-filter model directory")
	flag.Parse()

	model, err := privacyfilter.LoadModel(*modelDir)
	if err != nil {
		log.Fatalf("LoadModel: %v", err)
	}

	q := newQueue()

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	workerDone := make(chan struct{})
	go func() {
		defer close(workerDone)
		runWorker(ctx, q, model)
	}()

	staticRoot, err := fs.Sub(staticFS, "static")
	if err != nil {
		log.Fatalf("fs.Sub: %v", err)
	}

	mux := http.NewServeMux()
	mux.Handle("/", http.FileServerFS(staticRoot))
	mux.HandleFunc("POST /jobs", handleSubmit(q))
	mux.HandleFunc("GET /jobs/{id}", handleStatus(q))

	srv := &http.Server{
		Addr:              *addr,
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
	}

	go func() {
		log.Printf("listening on %s", *addr)
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Fatalf("ListenAndServe: %v", err)
		}
	}()

	<-ctx.Done()
	log.Printf("shutting down")
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_ = srv.Shutdown(shutdownCtx)
	q.close()
	<-workerDone
}
