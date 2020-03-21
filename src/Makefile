all: alexnet test alexnet_100

test:
	g++ -mavx -O3 test.cpp util.cpp -o test


alexnet:
	g++ -mavx -O3 alexnet.cpp util.cpp -o alexnet

alexnet_100:
	g++ -mavx -O3 alexnet_100.cpp util.cpp -o alexnet_100

clean:
	rm -f alexnet
	rm -f test
	rm -f alexnet_100

