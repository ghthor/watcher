devdaemon: main.go
	find . -name "$<" | entr -c -r go run $<

.PHONY: devdaemon
