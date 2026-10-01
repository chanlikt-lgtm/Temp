# Build the report:  make        (needs pdflatex/latexmk + the figures in ../figures)
#                    make clean
MAIN = wafer_report

pdf: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex
	latexmk -pdf -interaction=nonstopmode $(MAIN).tex || pdflatex -interaction=nonstopmode $(MAIN).tex

clean:
	latexmk -C $(MAIN) 2>/dev/null || true
	rm -f $(MAIN).aux $(MAIN).log $(MAIN).out $(MAIN).toc $(MAIN).fls $(MAIN).fdb_latexmk

.PHONY: pdf clean
