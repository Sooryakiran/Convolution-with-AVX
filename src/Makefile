all: alexnet test

test:
	g++ -mavx -O3 test.cpp util.cpp -o test


alexnet:
	g++ -mavx -O3 alexnet.cpp util.cpp -o alexnet

clean:
	rm -f alexnet
	rm -f test

