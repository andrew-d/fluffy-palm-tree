package main

import (
	"fmt"
	"log"
	"os"

	"github.com/andrew-d/openai-privacy"
)

func main() {
	model, err := privacyfilter.LoadModel("./model")
	if err != nil {
		log.Fatalf("LoadModel: %v", err)
	}

	texts := []string{
		"My name is Harry Potter and my email is harry.potter@hogwarts.edu.",
		"My name is Alice Smith",
		"Call me at 555-123-4567.",
	}
	if len(os.Args) > 1 {
		texts = os.Args[1:]
	}

	for _, text := range texts {
		fmt.Printf("input: %s\n", text)
		ents, err := model.Classify(text)
		if err != nil {
			log.Fatalf("Classify: %v", err)
		}
		for _, e := range ents {
			fmt.Printf("  [%s] %q  score=%.4f  offsets=[%d,%d]\n",
				e.EntityGroup, e.Word, e.Score, e.Start, e.End)
		}
		fmt.Println()
	}
}
